"""Paired 600-item medical MCQ evaluation, entirely local and resumable."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from data_processing.prepare_medical_holdout600 import ROOT,DATA,prepare
from experiments.run_causal_fact_experiment import LAB,load_base,available_commit_mib,read,save,sha

SYSTEM='Answer the medical multiple-choice question. Return only the single best option letter: A, B, C, or D.'
OLD=LAB/'outputs/cpt_medical_v1'
CPT=OLD/'cpt_seed42/final_adapter'
SFT=OLD/'sft_seed42/vqa_knowledge_case_context_attention_ffn/final_adapter'
DIRECT=LAB/'outputs/knowledge_experiments_v1/seed42/vqa_knowledge_case_context_attention_ffn/final_adapter'
PILOT=LAB/'outputs/causal_fact_v1'


def jobs():
    arms={'base':[], 'old_cpt':[CPT], 'old_sft':[DIRECT], 'old_cpt_sft':[CPT,SFT]}
    for seed in [42,43]:
        for scope in ['prose','single_question','multiple_questions']:
            arms[f'{scope}_seed{seed}']=[PILOT/f'{scope}_seed{seed}/checkpoints/000144/adapter']
    return arms


def prompt(row):
    return row['question']+'\n\n'+'\n'.join(f'{k}. {v}' for k,v in row['options'].items())


def parse_letter(text):
    strict=re.fullmatch(r'\s*([ABCD])\s*[.)]?\s*',text)
    explicit=re.match(r'^\s*(?i:(?:final\s+)?answer)\s*:\s*\*{0,2}([ABCDabcd])\*{0,2}(?:[.)]|\s*$)',text)
    if explicit is None:
        explicit=re.match(r'^\s*\*{0,2}([ABCD])\*{0,2}(?:[.)]|\s*$)',text)
    return (strict.group(1) if strict else None,explicit.group(1).upper() if explicit else None)


def plan():
    prepare()
    path=ROOT/'evaluation_plan.json'
    if path.exists():return read(path)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B',local_files_only=True)
    labels=[tok.encode(letter,add_special_tokens=False) for letter in 'ABCD']
    assert all(len(ids)==1 for ids in labels)
    rows=[json.loads(line) for line in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    lengths=[len(tok.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=prompt(r))],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)) for r in rows]
    assert max(lengths)<=1536,max(lengths)
    frozen={str(p):sha(p) for p in (LAB/'models/Qwen3.5-2B').iterdir() if p.suffix in ['.json','.safetensors','.jinja']}
    for paths in list(jobs().values())[:4]:
        for adapter in paths:
            for name in ['adapter_config.json','adapter_model.safetensors']:frozen[str(adapter/name)]=sha(adapter/name)
    value=dict(independent_questions=600,data_sha256=sha(DATA/'evaluation.jsonl'),system=SYSTEM,
        maximum_input_tokens=max(lengths),input_truncation=False,batch_size=4,max_new_tokens=32,
        thinking=False,decoding='greedy, no repetition penalty, original option order',letter_token_ids=[x[0] for x in labels],
        frozen_files=frozen,arms={k:[str(p) for p in v] for k,v in jobs().items()},
        primary='Argmax of first-step raw logits restricted to option-letter tokens A/B/C/D; medical choice selection under the frozen nonthinking prompt.',
        secondary=['Freely generated explicit leading answer letter; missing/ambiguous answer counted incorrect',
            'Strict single-letter format compliance','EOS termination and truncated outputs','Subject counts; paired repair/regression and bootstrap intervals'],
        interpretation='Constrained letter ranking and free generation reported separately. Neither is open-ended answer accuracy or reasoning validity. No LLM judge calls.',
        comparisons=[['base','old_cpt'],['base','old_sft'],['old_sft','old_cpt_sft'],['old_cpt','old_cpt_sft'],
            ['prose_seed42','single_question_seed42'],['single_question_seed42','multiple_questions_seed42'],
            ['prose_seed43','single_question_seed43'],['single_question_seed43','multiple_questions_seed43']],
        uncertainty='Per-question paired differences, subject-stratified bootstrap95%CI seed20260908 10000replicates; report repair/regression and exact two-sided McNemar p descriptively, without unadjusted multi-comparison significance claims.',
        label_review='Retain primary official labels; separately record suspected ambiguous/incorrect references. Do not drop disagreements after observing checkpoint scores.')
    save(path,value);return value


def validate():
    config=plan()
    assert sha(DATA/'evaluation.jsonl')==config['data_sha256']
    assert config['system']==SYSTEM
    for path,digest in config['frozen_files'].items():assert sha(Path(path))==digest,path
    return config


def evaluate_loaded(model,tok,arm,identity,smoke=False):
    import torch
    config=validate()
    out=ROOT/('smoke' if smoke else 'runs')/arm
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    else:save(out/'identity.json',identity)
    evaluator_hash=sha(Path(__file__))
    all_rows=[json.loads(line) for line in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    if smoke:all_rows=all_rows[:4]
    pending=[r for r in all_rows if not (out/'rows'/f"{r['source_id']}.json").exists()]
    model.eval();original_padding_side=tok.padding_side;tok.padding_side='left'
    checked=False
    for offset in range(0,len(pending),config['batch_size']):
        batch=pending[offset:offset+config['batch_size']]
        messages=[[dict(role='system',content=SYSTEM),dict(role='user',content=prompt(r))] for r in batch]
        texts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False) for m in messages]
        inputs=tok(texts,padding=True,return_tensors='pt',add_special_tokens=False).to(model.device)
        started=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(**inputs,do_sample=False,num_beams=1,repetition_penalty=1.0,
                max_new_tokens=config['max_new_tokens'],use_cache=True,return_dict_in_generate=True,
                output_logits=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            first=generated.logits[0][:,config['letter_token_ids']].float()
            if not checked:
                # Match Qwen3.5 generation's text+vision positions for left-padded text batches.
                positions=(inputs['attention_mask'].long().cumsum(-1)-1).masked_fill(inputs['attention_mask']==0,0)
                positions=positions.unsqueeze(0).expand(4,-1,-1)
                direct=model(**inputs,position_ids=positions,use_cache=True,logits_to_keep=1).logits[:,-1,config['letter_token_ids']].float()
                error=float((first-direct).abs().max())
                assert torch.allclose(first,direct,atol=.125,rtol=0),error
                save(out/'alignment_check.json',dict(max_letter_logit_difference=error,batch_size=len(batch)))
                checked=True
        for index,row in enumerate(batch):
            tokens=generated.sequences[index,inputs['input_ids'].shape[1]:].tolist()
            stopped=tok.eos_token_id in tokens
            if stopped:tokens=tokens[:tokens.index(tok.eos_token_id)+1]
            text=tok.decode(tokens,skip_special_tokens=True).strip()
            strict,explicit=parse_letter(text)
            logits=first[index].tolist();selected='ABCD'[max(range(4),key=lambda i:logits[i])]
            save(out/'rows'/f"{row['source_id']}.json",dict(source_id=row['source_id'],subject=row['subject'],
                evaluator_sha256=evaluator_hash,data_sha256=config['data_sha256'],
                reference_label=row['answer_label'],letter_logits=dict(zip('ABCD',logits)),ranked_letter=selected,
                ranked_correct=selected==row['answer_label'],strict_letter=strict,explicit_letter=explicit,
                strict_correct=strict==row['answer_label'],free_correct=explicit==row['answer_label'],
                prediction=text,generated_tokens=len(tokens),stopped_on_eos=stopped,
                batch_seconds=time.monotonic()-started,input_tokens=int(inputs['attention_mask'][index].sum())))
        done=len(all_rows)-len(pending)+offset+len(batch)
        save(ROOT/'status.json',dict(state='generating',arm=arm,completed=done,total=len(all_rows),worker_pid=os.getpid()))
        if done%40==0 or smoke:print('HOLDOUT',arm,done,'/',len(all_rows),flush=True)
        del generated,inputs,first
    records=[read(out/'rows'/f"{r['source_id']}.json") for r in all_rows]
    tok.padding_side=original_padding_side
    save(out/'complete.json',dict(n=len(records),ranked_correct=sum(r['ranked_correct'] for r in records),
        free_correct=sum(r['free_correct'] for r in records),strict_correct=sum(r['strict_correct'] for r in records),
        missing_explicit_letter=sum(r['explicit_letter'] is None for r in records),
        non_eos_outputs=sum(not r['stopped_on_eos'] for r in records)))


def run(arm,smoke=False):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    validate()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.40)
    paths=jobs()[arm]
    identity={str(p/name):sha(p/name) for p in paths for name in ['adapter_config.json','adapter_model.safetensors']}
    model=load_base()
    for index,path in enumerate(paths):
        model=PeftModel.from_pretrained(model,path)
        if arm in ['old_cpt','old_cpt_sft'] and index==0:
            model=model.merge_and_unload(safe_merge=True);del model.peft_config
    model.to('cuda')
    tok=AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B',local_files_only=True)
    evaluate_loaded(model,tok,arm,identity,smoke=smoke)
    del model;gc.collect();torch.cuda.empty_cache()


def pipeline(group='all'):
    validate()
    for arm,paths in jobs().items():
        if group=='historical' and arm not in ['base','old_cpt','old_sft','old_cpt_sft']:continue
        if (ROOT/'runs'/arm/'complete.json').exists():continue
        assert all((p/'adapter_model.safetensors').exists() for p in paths),arm
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            commit=available_commit_mib()
            if free>=7500 and commit>=8000:break
            save(ROOT/'status.json',dict(state='waiting_for_memory',gpu_free_mib=free,commit_free_mib=commit,worker_pid=os.getpid()))
            time.sleep(60)
        subprocess.run([sys.executable,'-u',__file__,'run','--arm',arm],cwd=LAB,check=True)
        subprocess.run([sys.executable,'-u',str(LAB/'experiments.analyze_medical_holdout600.py')],cwd=LAB,check=True)
    save(ROOT/'status.json',dict(state='group_generation_complete',group=group,
        analysis_file=str(ROOT/'results.json'),label_review_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','run','pipeline'])
    parser.add_argument('--arm',choices=list(jobs()),default='base')
    parser.add_argument('--group',choices=['historical','all'],default='all');args=parser.parse_args()
    try:
        if args.stage=='prepare':plan()
        elif args.stage=='pipeline':pipeline(args.group)
        else:run(args.arm,smoke=args.stage=='smoke')
    except Exception as error:
        save(ROOT/'status.json',dict(state='failed',error=repr(error),worker_pid=os.getpid()));raise
