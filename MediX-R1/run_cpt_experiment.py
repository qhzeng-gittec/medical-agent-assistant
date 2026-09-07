"""Frozen, paired CPT+SFT experiment using the existing four-task SFT recipe."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from data_io import load_jsonl, save_jsonl

LAB = Path(__file__).resolve().parent
ROOT = LAB/'outputs/cpt_medical_v1'
DATA = LAB/'data/cpt_medical_v1'
SOURCE = LAB/'data/knowledge_experiments_v1'
RECIPE = 'vqa_knowledge_case_context'
BASELINE = LAB/f'outputs/knowledge_experiments_v1/seed42/{RECIPE}_attention_ffn/final_adapter'
CPT = ROOT/'cpt_seed42/final_adapter'
SFT = ROOT/f'sft_seed42/{RECIPE}_attention_ffn/final_adapter'
ARMS = ['sft_only', 'cpt_sft']


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (ROOT/'plan.json').exists():
        return
    rows = sorted(load_jsonl(SOURCE/'test.jsonl'),key=lambda r:(r['task'],r['source_id']))
    train = load_jsonl(SOURCE/'train.jsonl')
    assert not {r['source_id'] for r in rows} & {r['source_id'] for r in train}
    DATA.mkdir(parents=True,exist_ok=True)
    save_jsonl(DATA/'evaluation.jsonl',[dict(r,cohort='cpt_frozen_project_test') for r in rows])
    save(ROOT/'plan.json',dict(frozen_before_new_training_and_generation=True,seed=42,
        arms=ARMS,sft_recipe=RECIPE,sft_train_sha256=sha(SOURCE/'recipes'/RECIPE/'train.jsonl'),
        baseline_adapter_sha256=sha(BASELINE/'adapter_model.safetensors'),
        evaluation_sha256=sha(DATA/'evaluation.jsonl'),counts=dict(Counter(r['task'] for r in rows)),
        primary='knowledge strict final-answer correctness (judge answer_score == 2)',
        significance='Two-sided exact McNemar p<0.05 AND paired question bootstrap 95% CI lower bound>0; report effect in percentage points.',
        practical_effect='At least +5 percentage points on the primary endpoint; statistical and practical results reported separately.',
        secondary=['partial-credit answer score','reasoning score','joint score','case','context','vqa'],
        generation=dict(decoding='greedy',enable_thinking=True,max_new_tokens=2048,reasoning_budget=1792,
                        text_batch_size=4,image_batch_size=1,image_max_edge=768),
        cpt=dict(rank=32,scope='language attention including linear-attention gates + FFN',learning_rate=5e-5,
                 medical_token_target=8000000,replay_fraction=.01,epochs=1,max_length=1024),
        sft=dict(rank=8,scope='attention_ffn',learning_rate=5e-5,epochs=1,batch_size=2,gradient_accumulation_steps=2,
                 image_max_edge=768,seed=42,fresh_adapter=True,frozen_cpt_base=True),
        limitations=['One trained seed: question-level significance is not across-training-seed robustness.',
                    'Previously used project holdout, not a newly sourced external benchmark.',
                    'Extra CPT adds data and compute; this estimates the pipeline effect, not a compute-matched objective ablation.',
                    'CPT is a small, selected guideline experiment on the post-trained Qwen3.5-2B, not reproduction of billion-token full-parameter CPT.',
                    'Medical judgments are model assessments, not clinician validation.']))
    print('Frozen 514 paired questions; primary endpoint: 150 medical knowledge questions.',flush=True)


def train():
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    assert sha(SOURCE/'recipes'/RECIPE/'train.jsonl') == plan['sft_train_sha256']
    jobs = [
      ('cpt',[sys.executable,'-u','train_cpt_lora.py'],CPT.parent),
      ('sft',[sys.executable,'-u','train_multitask_lora.py','--recipe',RECIPE,
        '--recipes-dir',str(SOURCE/'recipes'),'--output-root',str(ROOT/'sft_seed42'),
        '--base-adapter-dir',str(CPT),'--lora-scope','attention_ffn','--batch-size','2',
        '--gradient-accumulation-steps','2','--image-max-edge','768','--epochs','1',
        '--learning-rate','0.00005','--eval-steps','250','--seed','42','--gpu-memory-fraction','.8'],SFT.parent)]
    for stage,command,out in jobs:
        if (out/'training_complete.json').exists():
            continue
        checkpoints = sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints:
            command += ['--resume-from-checkpoint',str(checkpoints[-1])]
        save(ROOT/'status.json',dict(stage=stage,status='running',command=command))
        log = ROOT/f'{stage}.log'
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with log.open('a',encoding='utf-8') as stream:
            try:
                subprocess.run(command,cwd=LAB,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as error:
                save(ROOT/'status.json',dict(stage=stage,status='failed',returncode=error.returncode,log=str(log)))
                raise
        assert (out/'training_complete.json').exists()
    save(ROOT/'status.json',dict(stage='training',status='complete'))


def generate(arms):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from generate_reasoning_batch import generate_batch
    from evaluate_native_reasoning import generate as generate_one
    from native_reasoning import SYSTEM_PROMPT
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    assert sha(DATA/'evaluation.jsonl') == plan['evaluation_sha256']
    assert sha(BASELINE/'adapter_model.safetensors') == plan['baseline_adapter_sha256']
    rows = load_jsonl(DATA/'evaluation.jsonl')
    for arm in arms:
        folder = ROOT/'generation'/arm
        adapter = BASELINE if arm=='sft_only' else SFT
        batches = []
        for task in ['knowledge','case','context','vqa']:
            subset = [r for r in rows if r['task']==task]
            size = 1 if task=='vqa' else 4
            batches += [subset[i:i+size] for i in range(0,len(subset),size)]
        pending = [b for b in batches if not all((folder/'samples'/f"{r['source_id']}.json").exists() for r in b)]
        if not pending:
            continue
        processor = AutoProcessor.from_pretrained(LAB/'models/Qwen3.5-2B',do_resize=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(LAB/'models/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='sdpa')
        if arm=='cpt_sft':
            config = json.loads((SFT.parent/'run_config.json').read_text(encoding='utf-8'))
            assert config['base_adapter_sha256'] == sha(CPT/'adapter_model.safetensors')
            model = PeftModel.from_pretrained(model,CPT).merge_and_unload(safe_merge=True)
            del model.peft_config
        model = PeftModel.from_pretrained(model,adapter).to('cuda').eval()
        save(folder/'config.json',dict(arm=arm,adapter_sha256=sha(adapter/'adapter_model.safetensors'),
            cpt_adapter_sha256=sha(CPT/'adapter_model.safetensors') if arm=='cpt_sft' else None,
            evaluation_sha256=plan['evaluation_sha256'],protocol=plan['generation']))
        for index,batch in enumerate(pending,1):
            torch.manual_seed(42)
            if batch[0]['image_path']:
                responses = [generate_one(model,processor,batch[0],SYSTEM_PROMPT,2048,'greedy',1792,768)]
            else:
                responses = generate_batch(model,processor,batch)
            for row,response in zip(batch,responses,strict=True):
                save(folder/'samples'/f"{row['source_id']}.json",dict(response,source_id=row['source_id'],task=row['task']))
            save(ROOT/f'{arm}_generation_status.json',dict(batch=index,batches=len(pending),task=batch[0]['task']))
            print(arm,index,'/',len(pending),batch[0]['task'],'tokens',[r['generated_tokens'] for r in responses],flush=True)
        save(folder/'complete.json',dict(samples=len(rows)))
        del model,processor
        gc.collect()
        torch.cuda.empty_cache()


def score():
    import diagnose_reasoning_effects as diagnostic
    diagnostic.ROOT = ROOT
    rows = load_jsonl(DATA/'evaluation.jsonl')
    pending = [r for r in rows if not (ROOT/'diagnostics/questions'/f"{r['source_id']}.json").exists()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = []
        for row in pending:
            responses = {a:json.loads((ROOT/'generation'/a/'samples'/f"{row['source_id']}.json").read_text(encoding='utf-8')) for a in ARMS}
            jobs.append(pool.submit(diagnostic.score_question,row,responses,'diagnostics'))
        for i,job in enumerate(as_completed(jobs),1):
            result=job.result()
            print('judged',i,'/',len(jobs),result['source_id'],flush=True)
            save(ROOT/'scoring_status.json',dict(complete=len(rows)-len(jobs)+i,total=len(rows)))


def summarize():
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'diagnostics/questions').glob('*.json'))]
    assert len(rows)==sum(plan['counts'].values())
    groups = {}
    rng = np.random.default_rng(42)
    for task in ['knowledge','case','context','vqa']:
        valid = [r for r in rows if r['task']==task and r['reference_valid']]
        for row in valid:
            left,right = [row['models'][arm] for arm in ARMS]
            if left['final_answer'] == right['final_answer']:
                assert left['diagnosis']['answer_score'] == right['diagnosis']['answer_score'], row['source_id']
        group = dict(n=len(valid),excluded=[r['source_id'] for r in rows if r['task']==task and not r['reference_valid']],arms={})
        for arm in ARMS:
            diagnoses = [r['models'][arm]['diagnosis'] for r in valid]
            group['arms'][arm] = dict(accuracy=100*float(np.mean([d['answer_score']==2 for d in diagnoses])),
                **{m:50*float(np.mean([d[m] for d in diagnoses])) for m in ['answer_score','reasoning_score','joint_score']},
                errors=dict(Counter(f for d in diagnoses for f in d['flags'])),
                mean_generated_tokens=float(np.mean([r['models'][arm]['generated_tokens'] for r in valid])))
        a=np.array([r['models']['sft_only']['diagnosis']['answer_score']==2 for r in valid],dtype=int)
        b=np.array([r['models']['cpt_sft']['diagnosis']['answer_score']==2 for r in valid],dtype=int)
        delta=(b-a)*100
        bootstrap=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(axis=1)
        wins=int(((a==0)&(b==1)).sum()); losses=int(((a==1)&(b==0)).sum())
        p=float(binomtest(wins,wins+losses,.5,alternative='two-sided').pvalue) if wins+losses else 1.0
        group['paired_accuracy']=dict(delta_pp=float(delta.mean()),ci95_pp=np.quantile(bootstrap,[.025,.975]).tolist(),
             improved=wins,worsened=losses,mcnemar_exact_p=p)
        groups[task]=group
    primary=groups['knowledge']['paired_accuracy']
    significant=primary['delta_pp']>0 and primary['mcnemar_exact_p']<.05 and primary['ci95_pp'][0]>0
    result=dict(groups=groups,primary_significant_improvement=significant,
        primary_practical_improvement=primary['delta_pp']>=5,limitations=plan['limitations'],
        secondary_pvalues_are_exploratory=True,one_training_seed=True)
    save(ROOT/'results.json',result)
    save(ROOT/'status.json',dict(stage='complete',primary_significant_improvement=significant))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','generate','score','summarize','all'])
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    args=parser.parse_args()
    if args.stage=='all':
        prepare(); train(); generate(args.arms); score(); summarize()
    elif args.stage=='generate':
        generate(args.arms)
    else:
        globals()[args.stage]()
