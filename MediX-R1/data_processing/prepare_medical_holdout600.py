"""Freeze a new local-project MCQ holdout without looking at model predictions."""
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re
import unicodedata

from experiments.run_causal_fact_experiment import LAB,read,save,sha

ROOT=LAB/'outputs/medical_holdout600_v1'
DATA=LAB/'data/medical_holdout600_v1'
SEED=20260908
UUID=re.compile(r'[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}',re.I)


def normalize(text):
    return re.sub(r'[^a-z0-9]+',' ',unicodedata.normalize('NFKC',text).lower()).strip()


def prepare():
    import pyarrow.parquet as pq
    if (ROOT/'data_plan.json').exists():
        plan=read(ROOT/'data_plan.json')
        assert sha(DATA/'evaluation.jsonl')==plan['evaluation_sha256']
        return
    raw=LAB/'data/medmcqa_raw/validation.parquet'
    source=pq.read_table(raw).to_pylist()
    names={'train.jsonl','validation.jsonl','test.jsonl','evaluation.jsonl',
        'train_candidates.jsonl','validation_candidates.jsonl','test_candidates.jsonl'}
    files=sorted(p for p in (LAB/'data').rglob('*.jsonl') if p.name in names and DATA not in p.parents)
    seen_ids=set();seen_questions=set();scanned={}
    for path in files:
        scanned[str(path)]=sha(path)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                for key in ['id','source_id','source_group','family_id']:
                    seen_ids.update(x.lower() for x in UUID.findall(str(row.get(key,''))))
                for key in ['original_normalized_question','normalized_question','question','user_text']:
                    if row.get(key):seen_questions.add(normalize(row[key]))
    # Exact source IDs recover original wording even when local training used rewritten questions.
    for r in source:
        if r['id'].lower() in seen_ids:seen_questions.add(normalize(r['question']))
    training_raw=LAB/'data/medmcqa_raw/train.parquet'
    for r in pq.read_table(training_raw,columns=['id','question']).to_pylist():
        if r['id'].lower() in seen_ids:seen_questions.add(normalize(r['question']))
    index=defaultdict(set)
    texts=[]
    def add_text(text):
        key=len(texts);texts.append(text)
        words=text.split()
        for pair in set(zip(words,words[1:])):index[pair].add(key)
    for q in sorted(seen_questions):add_text(q)
    def duplicate(q):
        if q in seen_questions:return 'exact_question'
        words=q.split();matches=Counter()
        for pair in set(zip(words,words[1:])):matches.update(index.get(pair,()))
        for key,count in matches.items():
            other=texts[key]
            if count>=2 and min(len(q),len(other))/max(len(q),len(other))>=.8:
                if SequenceMatcher(None,q,other,autojunk=False).ratio()>=.9:return 'near_question'
        return None
    rng=random.Random(SEED)
    ordered=sorted(source,key=lambda r:r['id']);rng.shuffle(ordered)
    pool=[];excluded=[]
    for row in ordered:
        opts=[row[k] for k in ['opa','opb','opc','opd']]
        question=row['question'] or '';norm=normalize(question)
        reason=None
        if row['id'].lower() in seen_ids:reason='prior_local_source_id'
        elif row['choice_type']!='single' or row['cop'] not in range(4):reason='not_single_labeled'
        elif not all(isinstance(o,str) and o.strip() for o in opts):reason='missing_option'
        elif len(set(normalize(o) for o in opts))!=4:reason='duplicate_options'
        elif len(norm)<20 or len(question)>3000:reason='question_length'
        elif re.search(r'\b(image|figure|diagram|photograph|shown|illustrated)\b',question,re.I):reason='possible_missing_visual'
        elif not row['exp'] or len(str(row['exp']).split())<8:reason='insufficient_source_explanation'
        else:reason=duplicate(norm)
        if reason:
            excluded.append(dict(id=row['id'],reason=reason));continue
        pool.append(dict(source_id='medmcqa-'+row['id'],source_split='official_validation',
            question=question,options=dict(zip('ABCD',opts)),answer_label='ABCD'[row['cop']],
            reference=opts[row['cop']],reference_explanation=row['exp'],subject=row['subject_name'] or 'Unspecified',
            normalized_question=norm,label_provenance='Original MedMCQA label; not clinician revalidated.'))
        add_text(norm);seen_questions.add(norm)
    assert len(pool)>=600,len(pool)
    groups=defaultdict(list)
    for r in pool:groups[r['subject']].append(r)
    quotas={s:int(len(rs)*600/len(pool)) for s,rs in groups.items()}
    remaining=600-sum(quotas.values())
    for s in sorted(groups,key=lambda s:(-(len(groups[s])*600/len(pool)-quotas[s]),s))[:remaining]:quotas[s]+=1
    selected=[]
    for s in sorted(groups):
        rng.shuffle(groups[s]);selected.extend(groups[s][:quotas[s]])
    selected.sort(key=lambda r:r['source_id'])
    assert len(selected)==len({r['source_id'] for r in selected})==600
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'evaluation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    save(DATA/'exclusions.json',excluded)
    save(ROOT/'data_plan.json',dict(seed=SEED,independent_questions=600,
        evaluation_sha256=sha(DATA/'evaluation.jsonl'),source_sha256=sha(raw),training_raw_sha256=sha(training_raw),
        excluded_input_hashes=scanned,eligible_pool=len(pool),source_rows=len(source),prior_local_ids=len(seen_ids),
        subject_quotas=quotas,exclusions=dict(Counter(r['reason'] for r in excluded)),
        selection='Proportional subject stratification from a shuffled eligible pool, largest-remainder quotas. No model predictions used.',
        deduplication='Exact IDs and normalized questions; >=2 shared word bigrams, character-length ratio>=.8, SequenceMatcher>=.9 against scanned local sets and earlier eligible rows.',
        limitations=['New local-project holdout, not guaranteed absent from the foundation model pretraining.',
            'Official MCQ labels and explanations are retained; ambiguity or label issues must be audited and reported without changing frozen primary labels.',
            'Lexical near-duplicate exclusion is not complete semantic decontamination.',
            'Measures medical multiple-choice selection, not open-ended clinical answering or reasoning validity.',
            'Small subject strata are descriptive; overall600 does not imply hundreds per subject.']))
    print(json.dumps(read(ROOT/'data_plan.json')|{'excluded_input_hashes':len(scanned)},ensure_ascii=False),flush=True)


if __name__=='__main__':prepare()
