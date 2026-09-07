"""Run the frozen T0-T3 continuation pilot and blind development evaluation."""
import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from data_io import load_jsonl
from diagnose_reasoning_effects import RUBRIC, SCORE_SCHEMA, obj, response_id
from medical_teacher import call_teacher
from native_reasoning import SYSTEM_PROMPT
from prepare_format_factorial import DATA, ROOT, INITIAL, MODEL, save, sha

LAB = Path(__file__).resolve().parent
ARMS = ['C_start','T0','T1','T2','T3']
FORMAT_RUBRIC = '''Also classify complete_sentence separately from correctness: true when the final answer is a grammatical standalone declarative sentence with a finite verb, or multiple such sentences. An isolated name, number, list, noun phrase or participial phrase is false. A sentence can be fully wrong and still complete_sentence=true. Do not subtract correctness points for a concise but semantically sufficient answer fragment. Do not reward full sentences with more correctness points. Score semantic content, not target wording or length. Assign the same semantic_answer_key to semantically equivalent final answers on this question, regardless of sentence framing; such answers MUST receive the same answer_score. Different factual claims or materially different scope need different keys. Do not merge answers just because they earn the same score.'''
ITEM_SCHEMA = json.loads(json.dumps(SCORE_SCHEMA))
ITEM_SCHEMA['properties']['id'] = {'type':'string'}
ITEM_SCHEMA['required'].append('id')
CANDIDATE = ITEM_SCHEMA['properties']['candidates']['items']
CANDIDATE['properties']['complete_sentence'] = {'type':'boolean'}
CANDIDATE['required'].append('complete_sentence')
CANDIDATE['properties']['semantic_answer_key'] = {'type':'string'}
CANDIDATE['required'].append('semantic_answer_key')
BATCH_SCHEMA = obj({'items':{'type':'array','items':ITEM_SCHEMA}})


def plan():
    return json.loads((DATA/'experiment_plan.json').read_text(encoding='utf-8'))


def train(arms):
    frozen=plan()
    assert sha(INITIAL/'adapter_model.safetensors') == frozen['initial_adapter_sha256']
    for arm in arms:
        assert arm in ['T0','T1','T2','T3']
        directory=ROOT/'training'/f'{arm}_attention'
        marker=directory/'training_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['global_step']==frozen['optimizer_steps']
            continue
        command=[sys.executable,'-u','train_multitask_lora.py','--recipes-dir',str(DATA/'recipes'),
                 '--recipe',arm,'--output-root',str(ROOT/'training'),'--initial-adapter-dir',str(INITIAL),
                 '--lora-scope','attention','--batch-size','1','--gradient-accumulation-steps','4',
                 '--epochs','2','--learning-rate','0.00002','--eval-steps','100',
                 '--seed','42','--image-max-edge','512','--gpu-memory-fraction','0.4']
        checkpoints=sorted(directory.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
        if checkpoints:
            # Resume an interrupted arm with its optimizer and RNG state, never fresh continuation.
            i=command.index('--initial-adapter-dir')
            del command[i:i+2]
            command+=['--resume-from-checkpoint',str(checkpoints[0])]
        log=ROOT/'logs'/f'train_{arm}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        save(ROOT/'training_status.json',dict(stage='training',arm=arm,command=command))
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=LAB,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        if result.returncode:
            raise RuntimeError(f'Training failed for {arm}; see {log}')
        assert json.loads(marker.read_text())['global_step']==frozen['optimizer_steps']
        config=json.loads((directory/'run_config.json').read_text())
        assert config['recipe_sha256']==frozen['manifests'][arm]['train_sha256']
        print(f'{arm} training complete: {frozen["optimizer_steps"]} steps',flush=True)
    assert sha(INITIAL/'adapter_model.safetensors') == frozen['initial_adapter_sha256']
    save(ROOT/'training_status.json',dict(stage='complete',arms=arms))


def generate(arms):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from generate_reasoning_batch import generate_batch
    from evaluate_native_reasoning import generate as generate_one

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.4)
    frozen=json.loads((DATA/'plan.json').read_text(encoding='utf-8'))
    assert sha(DATA/'evaluation.jsonl')==frozen['evaluation_sha256']
    rows=load_jsonl(DATA/'evaluation.jsonl')
    for arm in arms:
        adapter=INITIAL if arm=='C_start' else ROOT/'training'/f'{arm}_attention/final_adapter'
        if arm!='C_start':
            assert json.loads((adapter.parent/'training_complete.json').read_text())['global_step']==plan()['optimizer_steps']
        config=dict(arm=arm,adapter_sha256=sha(adapter/'adapter_model.safetensors'),
                    evaluation_sha256=frozen['evaluation_sha256'],max_new_tokens=2048,
                    reasoning_budget=1792,decoding='greedy',image_max_edge=512,enable_thinking=True)
        directory=ROOT/'generation'/arm
        if (directory/'config.json').exists():
            assert json.loads((directory/'config.json').read_text())==config
        save(directory/'config.json',config)
        if (directory/'complete.json').exists():
            continue
        processor=AutoProcessor.from_pretrained(LAB/'models/Qwen3.5-2B',do_resize=False)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(LAB/'models/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(model,adapter).to('cuda').eval()
        batches=[]
        text=[r for r in rows if r['task']=='knowledge']
        batches += [text[i:i+4] for i in range(0,len(text),4)]
        batches += [[r] for r in rows if r['task']=='vqa']
        for index,batch in enumerate(batches,1):
            if all((directory/'samples'/f"{r['source_id']}.json").exists() for r in batch):
                continue
            responses=(generate_batch(model,processor,batch) if batch[0]['task']=='knowledge'
                       else [generate_one(model,processor,batch[0],SYSTEM_PROMPT,2048,'greedy',1792,512)])
            for row,response in zip(batch,responses,strict=True):
                response.update(source_id=row['source_id'],task=row['task'])
                save(directory/'samples'/f"{row['source_id']}.json",response)
            print(f'{arm}: generation batch {index}/{len(batches)}',flush=True)
        assert len(list((directory/'samples').glob('*.json')))==len(rows)
        save(directory/'complete.json',dict(samples=len(rows),config=config))
        del model,processor
        gc.collect()
        torch.cuda.empty_cache()


def score_batch(rows):
    payload=[]
    all_responses={}
    for row in rows:
        responses={arm:json.loads((ROOT/'generation'/arm/'samples'/f"{row['source_id']}.json").read_text(encoding='utf-8')) for arm in ARMS}
        all_responses[row['source_id']]=responses
        candidates=list({response_id(r):dict(id=response_id(r),reasoning=r['reasoning'],final_answer=r['final_answer']) for r in responses.values()}.values())
        random.Random(42).shuffle(candidates)
        payload.append(dict(id=row['source_id'],task=row['task'],question=row['user_text'],reference=row['reference'],
                            reference_explanation=row['reference_explanation'],candidates=candidates))
    prompt=RUBRIC+'\n'+FORMAT_RUBRIC+'\nReturn one item per question ID.\n\n'+json.dumps(payload,ensure_ascii=False)
    key=hashlib.sha256((prompt+json.dumps(BATCH_SCHEMA)).encode()).hexdigest()
    path=ROOT/'judge/calls'/f'{key}.json'
    result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else call_teacher(prompt,[Path(r['image_path']) for r in rows if r['image_path']],BATCH_SCHEMA,path,MODEL)
    assert len(result['items'])==len(rows) and {r['id'] for r in result['items']}=={r['source_id'] for r in rows}
    for item in result['items']:
        responses=all_responses[item['id']]
        expected={response_id(r) for r in responses.values()}
        assert len(item['candidates'])==len(expected) and {r['id'] for r in item['candidates']}==expected
        scores={r['id']:r for r in item['candidates']}
        for value in scores.values():
            assert set(value['flags'])=={f['type'] for f in value['findings']}
            value['joint_score']=min(value['answer_score'],value['reasoning_score'])
        save(ROOT/'judge/questions'/f"{item['id']}.json",dict(source_id=item['id'],reference_valid=item['reference_valid'],
            reference_comment=item['reference_comment'],call_sha256=key,
            models={arm:dict(response,diagnosis=scores[response_id(response)]) for arm,response in responses.items()}))


def score():
    rows=load_jsonl(DATA/'evaluation.jsonl')
    assert all((ROOT/'generation'/a/'complete.json').exists() for a in ARMS)
    text=[r for r in rows if r['task']=='knowledge']
    batches=[text[i:i+3] for i in range(0,len(text),3)]+[[r] for r in rows if r['task']=='vqa']
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(score_batch,batch) for batch in batches]
        for index,future in enumerate(as_completed(futures),1):
            future.result()
            save(ROOT/'score_status.json',dict(completed_batches=index,total_batches=len(futures)))
            print(f'Scored {index}/{len(futures)} batches',flush=True)


def summarize():
    frozen=plan()
    rows=load_jsonl(DATA/'evaluation.jsonl')
    questions={r['source_id']:json.loads((ROOT/'judge/questions'/f"{r['source_id']}.json").read_text(encoding='utf-8')) for r in rows}
    groups={}
    conflicts=[]
    for question in questions.values():
        finals={}
        meanings={}
        for arm,response in question['models'].items():
            prior=finals.setdefault(response['final_answer'],response['diagnosis']['answer_score'])
            if prior!=response['diagnosis']['answer_score']:
                conflicts.append(dict(source_id=question['source_id'],arm=arm))
            prior=meanings.setdefault(response['diagnosis']['semantic_answer_key'],response['diagnosis']['answer_score'])
            if prior!=response['diagnosis']['answer_score']:
                conflicts.append(dict(source_id=question['source_id'],arm=arm,kind='semantic_equivalence'))
    for task in ['knowledge','vqa']:
        ids=[r['source_id'] for r in rows if r['task']==task and questions[r['source_id']]['reference_valid']]
        for arm in ARMS:
            responses=[questions[sid]['models'][arm] for sid in ids]
            groups[f'{task}/{arm}']=dict(n=len(ids),
                **{m:50*statistics.mean(r['diagnosis'][m] for r in responses) for m in ['answer_score','reasoning_score','joint_score']},
                full_sentence_percent=100*statistics.mean(r['diagnosis']['complete_sentence'] for r in responses),
                median_answer_words=statistics.median(len(r['final_answer'].split()) for r in responses),
                median_reasoning_words=statistics.median(len(r['reasoning'].split()) for r in responses),
                reasoning_material_error_percent=100*statistics.mean(r['diagnosis']['reasoning_score']==0 for r in responses),
                forced_ends=sum(r['forced_reasoning_end'] for r in responses),
                length_stops=sum(r['stop_reason']=='length' for r in responses))
    contrasts={}
    weights={key:{key.split('-')[0]:1,key.split('-')[1]:-1} for key in frozen['main_contrasts']+['T0-C_start']}
    weights.update(form={'T1':.5,'T3':.5,'T0':-.5,'T2':-.5},relation={'T2':.5,'T3':.5,'T0':-.5,'T1':-.5},interaction={'T3':1,'T2':-1,'T1':-1,'T0':1})
    for task in ['knowledge','vqa']:
        valid=[r for r in rows if r['task']==task and questions[r['source_id']]['reference_valid']]
        strata={}
        for i,row in enumerate(valid):
            strata.setdefault(row.get('subject','vqa'),[]).append(i)
        for name,weight in weights.items():
            result={}
            for metric in ['answer_score','reasoning_score','joint_score','complete_sentence']:
                scale=100 if metric=='complete_sentence' else 50
                diffs=np.array([scale*sum(w*questions[r['source_id']]['models'][arm]['diagnosis'][metric] for arm,w in weight.items()) for r in valid])
                rng=np.random.default_rng(42)
                bootstrap=sum(diffs[ix][rng.integers(0,len(ix),size=(20000,len(ix)))].sum(axis=1) for ix in strata.values())/len(valid)
                result[metric]=dict(delta=float(diffs.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                                    improved=int((diffs>0).sum()),worsened=int((diffs<0).sum()),unchanged=int((diffs==0).sum()))
            contrasts[f'{task}/{name}']=result
    for arm in ['T0','T1','T2','T3']:
        config=json.loads((ROOT/'training'/f'{arm}_attention/run_config.json').read_text())
        assert config['recipe_sha256']==frozen['manifests'][arm]['train_sha256']
        assert config['initial_adapter_sha256']==frozen['initial_adapter_sha256']
    assert sha(INITIAL/'adapter_model.safetensors')==frozen['initial_adapter_sha256']
    result=dict(plan=frozen,groups=groups,contrasts=contrasts,invalid_questions=[sid for sid,q in questions.items() if not q['reference_valid']],
                same_final_score_conflicts=conflicts,nominal_intervals=True,training_seed_count=1,
                status='needs_score_review' if conflicts else 'complete')
    save(ROOT/'results.json',result)
    print(json.dumps(dict(groups=groups,conflicts=conflicts),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','generate','score','summarize'])
    parser.add_argument('--arms',nargs='+',choices=ARMS)
    args=parser.parse_args()
    if args.phase=='train':
        train(args.arms or ARMS[1:])
    elif args.phase=='generate':
        generate(args.arms or ARMS)
    elif args.phase=='score':
        score()
    else:
        summarize()
