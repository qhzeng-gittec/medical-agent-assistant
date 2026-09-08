"""Native Windows diagnostic inference with length-bucketed batches and bounded thinking."""
import argparse
import gc
import hashlib
import time
from collections import Counter

import experiments.run_cpt_diagnosis as diagnostic
from common.generation import generate_batch

ROOT = diagnostic.LAB / 'outputs/cpt_diagnosis_v3'
diagnostic.ROOT = ROOT
read, save, sha = diagnostic.read, diagnostic.save, diagnostic.sha


def prepare():
    if (ROOT/'native_plan.json').exists():
        return
    diagnostic.prepare()
    plan=read(ROOT/'plan.json')
    plan['generation']=dict(max_new_tokens=768,reasoning_budget=512,thinking=True,
        sampling_profile='qwen35',temperature=1.,top_p=.95,top_k=20,presence_penalty=1.5,
        max_batch_size=16,max_padded_input_tokens=6144,backend='native_windows_transformers',
        seed_policy='SHA256 of ordered batch input hashes and trial seed; deterministic length buckets. Batch composition affects stochastic outputs.')
    plan['scoring']='Blinded paired judging of newly generated responses from all four stages, within the same bounded decoding protocol.'
    plan['limitations'] += ['Budget and sampling changed together after observing repetition. Do not attribute cross-protocol changes to one factor.',
        'Primary answers are stochastic single draws, not greedy; extra four draws are conditional oracle coverage.',
        'Fixed batches sorted by input length, not continuous batching. Shortened reasoning budget can affect reasoning quality.',
        'Resume preserves complete deterministic batches; batch composition and ordering are fixed before inference.']
    save(ROOT/'plan.json',plan)
    save(ROOT/'native_plan.json',dict(plan_sha256=sha(ROOT/'plan.json'),expected_outputs=1504,expected_questions=247))
    save(diagnostic.LAB/'outputs/cpt_diagnosis_v1/status.json',dict(status='stopped',superseded_by='cpt_diagnosis_v3',reason='Repetitive long greedy outputs; native Windows bounded batch evaluation replaces this incomplete run.'))


def verify_inputs():
    plan=read(ROOT/'plan.json')
    assert sha(ROOT/'plan.json')==read(ROOT/'native_plan.json')['plan_sha256']
    assert sha(ROOT/'evaluation.jsonl')==plan['evaluation_sha256']
    assert sha(ROOT/'evidence.json')==plan['evidence_sha256']
    for path,expected in plan['models']['base'].items():
        assert sha(diagnostic.LAB/path)==expected
    for key,folder in [('cpt',diagnostic.CPT),('sft',diagnostic.SFT),('baseline',diagnostic.BASELINE)]:
        assert sha(folder/'adapter_model.safetensors')==plan['models'][key]


def batches(jobs):
    pending=[]
    for job in sorted(jobs,key=lambda j:(j['input_tokens'],j['source_id'])):
        if pending and (len(pending)==16 or job['input_tokens']*(len(pending)+1)>6144):
            yield pending
            pending=[]
        pending.append(job)
    if pending:
        yield pending


def generate(benchmark=False):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from common.native_reasoning import SYSTEM_PROMPT
    verify_inputs()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    plan,evidence=read(ROOT/'plan.json'),read(ROOT/'evidence.json')
    rows=diagnostic.load_jsonl(ROOT/'evaluation.jsonl')
    for arm in (['base'] if benchmark else diagnostic.ARMS):
        processor=AutoProcessor.from_pretrained(diagnostic.LAB/'models/Qwen3.5-2B',do_resize=False)
        groups={}
        for row in rows:
            for variant,prompt,seed in diagnostic.variants(row,arm,plan,evidence):
                messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':[{'type':'text','text':prompt['user_text']}]}]
                ids=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True)
                groups.setdefault((variant,42 if seed is None else seed),[]).append(dict(prompt,
                    input_tokens=len(ids[0]),input_sha256=hashlib.sha256(prompt['user_text'].encode()).hexdigest()))
        work=[(variant,seed,batch) for (variant,seed),jobs in groups.items() for batch in batches(jobs)]
        if not benchmark and all((ROOT/'generation'/variant/(j['source_id']+'.json')).exists() for variant,_,batch in work for j in batch):
            continue
        model=Qwen3_5ForConditionalGeneration.from_pretrained(diagnostic.LAB/'models/Qwen3.5-2B',dtype=torch.bfloat16,attn_implementation='sdpa')
        if arm in ['cpt_only','cpt_sft']:
            model=PeftModel.from_pretrained(model,diagnostic.CPT).merge_and_unload(safe_merge=True)
            del model.peft_config
        if arm in ['sft_only','cpt_sft']:
            model=PeftModel.from_pretrained(model,diagnostic.BASELINE if arm=='sft_only' else diagnostic.SFT)
        model.to('cuda').eval()
        if benchmark:
            sample=sorted(groups[('base__native',42)],key=lambda j:j['input_tokens'])[:16]
            generate_batch(model,processor,sample[:1],max_new_tokens=32,reasoning_budget=16,sampling_profile='qwen35')
            results=[]
            for size in [4,16]:
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                outputs=[]
                for offset in range(0,16,size):
                    torch.manual_seed(42)
                    outputs+=generate_batch(model,processor,sample[offset:offset+size],max_new_tokens=192,reasoning_budget=128,sampling_profile='qwen35')
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                assert len(outputs)==16 and all(r['generated_tokens']<=192 for r in outputs)
                metrics=dict(batch_size=size,requests=16,seconds=elapsed,generated_tokens=sum(r['generated_tokens'] for r in outputs),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    complete=sum(r['response_complete'] for r in outputs))
                metrics['tokens_per_second']=metrics['generated_tokens']/elapsed
                results.append(metrics)
                save(ROOT/'benchmark'/f'batch{size}.json',dict(metrics=metrics,outputs=outputs))
                print(metrics,flush=True)
            save(ROOT/'benchmark/results.json',dict(measurements=results,scope='Same 16 shortest input questions; 192/128 budget for throughput only; stochastic results can vary with batch size.'))
        else:
            start=time.perf_counter()
            done=0
            for variant,trial_seed,batch in work:
                paths=[ROOT/'generation'/variant/(j['source_id']+'.json') for j in batch]
                if all(p.exists() for p in paths):
                    continue
                # Regenerate a partial batch with its original composition and seed; retain completed files.
                batch_seed=int.from_bytes(hashlib.sha256(('|'.join(j['input_sha256'] for j in batch)+':'+str(trial_seed)).encode()).digest()[:4],'big')
                torch.manual_seed(batch_seed)
                save(ROOT/'status.json',dict(status='running',stage='generation',arm=arm,variant=variant,
                    batch_size=len(batch),completed_this_run=done,elapsed_seconds=time.perf_counter()-start))
                responses=generate_batch(model,processor,batch,max_new_tokens=768,reasoning_budget=512,sampling_profile='qwen35')
                for job,response,path in zip(batch,responses,paths,strict=True):
                    assert response['generated_tokens']<=768
                    if not path.exists():
                        save(path,dict(response,source_id=job['source_id'],task=job['task'],variant=variant,
                            input_sha256=job['input_sha256'],seed=batch_seed,trial_seed=trial_seed))
                        done+=1
                print(arm,variant,'batch',len(batch),'completed',done,'tokens',[r['generated_tokens'] for r in responses],flush=True)
            save(ROOT/'generation_metrics'/(arm+'.json'),dict(new_responses=done,seconds=time.perf_counter()-start))
        del model,processor
        gc.collect()
        torch.cuda.empty_cache()
    if not benchmark:
        save(ROOT/'generation_complete.json',dict(complete=True))


def score():
    import experiments.diagnose_reasoning_effects as judge
    judge.TEACHER=read(ROOT/'scoring_model_amendment.json')['remaining_judge']
    diagnostic.score()


def summarize():
    diagnostic.summarize()
    result=read(ROOT/'results.json')
    result.pop('judge_drift')
    result['cross_protocol_note']='New decoding outputs cannot be interpreted as pure judge drift; compare v3 stages together.'
    for item in result['sampling']:
        item['primary_draw_correct']=item.pop('greedy_correct')
    amendment=read(ROOT/'scoring_model_amendment.json')
    questions=[read(p) for p in (ROOT/'diagnostics/questions').glob('*.json')]
    old=amendment['completed_questions']
    for sid,metadata in old.items():
        assert sha(ROOT/'diagnostics/questions'/(sid+'.json'))==metadata['sha256']
    judge_counts=Counter(q.get('judge_model',old.get(q['source_id'],{}).get('judge_model')) for q in questions)
    assert None not in judge_counts
    result['scoring_models']=dict(counts=dict(judge_counts),
        by_task={task:dict(Counter(q.get('judge_model',old.get(q['source_id'],{}).get('judge_model')) for q in questions if q['task']==task)) for task in ['knowledge','context']},
        note='User requested a mid-run judge change. Completed judgments were preserved byte-for-byte; every question retains a common judge across model arms. Different judges may have different calibration.')
    save(ROOT/'results.json',result)
    verify_inputs()
    files=list((ROOT/'generation').glob('*/*.json'))
    assert len(files)==1504
    counts=Counter(p.parent.name for p in files)
    for arm in diagnostic.ARMS:
        assert counts[arm+'__native']==247 and counts[arm+'__concise']==83 and counts[arm+'__evidence']==23
    assert all(counts['cpt_sft__sample_'+str(s)]==23 for s in [101,102,103,104])
    assert len(list((ROOT/'diagnostics/questions').glob('*.json')))==247
    save(ROOT/'final_verification.json',dict(outputs=1504,questions=247,inputs_unchanged=True,complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','benchmark','generate','score','summarize','all'])
    args=parser.parse_args()
    prepare()
    if args.stage=='all': generate(); score(); summarize()
    elif args.stage=='benchmark': generate(benchmark=True)
    elif args.stage=='generate': generate()
    elif args.stage=='score': score()
    elif args.stage=='summarize': summarize()
