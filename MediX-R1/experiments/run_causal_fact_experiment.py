"""Matched-supervision prose versus question-conditioned fact learning."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# Keep weight loading sequential to limit transient memory use on Windows.
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'

from data_processing.causal_fact_data import FACTS, CONTROL_IDS

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB/'outputs/causal_fact_v1'
DATA = LAB/'data/causal_fact_v1'
SYSTEM = 'Answer the medical knowledge question accurately. Give only the requested term or phrase.'
SCOPES = ['prose','single_question','multiple_questions']
SEEDS = [42,43]
STEPS = [24,72,144]
TARGETS = (r'model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn)\.'
           r'(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)'
           r'|model\.language_model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def prefix_ids(tokenizer, question):
    return tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=question)],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)


def prepare():
    if (ROOT/'plan.json').exists():
        return
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B',local_files_only=True)
    rows=[]
    for sid,prefix,answer,distractors,training,heldout in FACTS:
        assert len(training)==3 and len(heldout)==2 and not set(training)&set(heldout)
        assert len(set([answer]+distractors))==4
        paragraph=prefix+' '+answer+'.'
        ids=tok.encode(paragraph,add_special_tokens=False)
        raw_prefix=tok.encode(prefix,add_special_tokens=False)
        assert ids[:len(raw_prefix)]==raw_prefix, sid
        rows.append(dict(id=sid,partition='unexposed_control' if sid in CONTROL_IDS else 'trained',
            prefix=prefix,answer=answer,distractors=distractors,paragraph=paragraph,
            training_questions=training,heldout_questions=heldout,
            supervised_ids=ids+[tok.eos_token_id],raw_prefix_ids=raw_prefix,
            question_prefix_ids=[prefix_ids(tok,q) for q in training]))
    DATA.mkdir(parents=True,exist_ok=True)
    save(DATA/'facts.json',rows)
    frozen={str(LAB/'data_processing/causal_fact_data.py'):sha(LAB/'data_processing/causal_fact_data.py'),str(DATA/'facts.json'):sha(DATA/'facts.json')}
    for p in (LAB/'models/Qwen3.5-2B').glob('*'):
        if p.is_file() and p.suffix in ['.json','.safetensors','.jinja']:
            frozen[str(p)]=sha(p)
    old=LAB/'data/knowledge_experiments_v1/recipes/vqa_knowledge_case_context/train.jsonl'
    frozen[str(old)]=sha(old)
    save(ROOT/'plan.json',dict(scopes=SCOPES,seeds=SEEDS,checkpoints=STEPS,
        trained_facts=12,unexposed_controls=4,rank=32,alpha=64,dropout=.05,
        learning_rate=5e-5,optimizer='AdamW',weight_decay=.01,max_grad_norm=1.,
        micro_batch=1,accumulation=4,gpu_memory_fraction=.40,frozen_files=frozen,
        max_training_sequence=max(len(p)+len(r['supervised_ids']) for r in rows for p in r['question_prefix_ids']),
        supervision='Identical paragraph target and EOS token IDs, identical fact schedule, equal supervised tokens and fact exposures in all three arms. Raw prose has an EOS document-boundary prefix; QA uses a masked chat question prefix. Multi-question cycles three question forms; single-question always uses form zero.',
        exposure_schedule={str(s):s*4//12 for s in STEPS},
        primary='Gold versus three prespecified medical distractors: both sum log-probability and length-normalized log-probability rank/margin. Compare trained raw continuation, trained question and two held-out questions at every checkpoint. Same reference candidates across checkpoints.',
        secondary='Greedy free generation, max48 tokens, no thinking. Report exact normalized answer separately from gold-only keyword matches and review disagreements. Four untrained facts track spillover.',
        decisions=[
            'Raw continuation improves consistently across seeds but heldout QA does not: evidence of learned continuation failing to transfer to question-conditioned recall.',
            'Multiple-question beats single-question on heldout QA at matched exposure and supervised tokens: evidence supporting diversity of access patterns.',
            'Even exposed continuation fails to improve despite falling loss: inspect optimization and target ranking before claiming a capacity ceiling.',
            'After acquisition, original SFT is tested as a separate intervention with fresh rank8 attention-only versus attention+FFN adapters. If enabled SFT reduces retrieval and disabling it restores the frozen source model, that demonstrates functional interference rather than physical erasure.'
        ],
        retention_plan=dict(source='Prespecified prose seed42 step144, whether successful or not; quantify only facts acquired at that checkpoint for retention, report others separately.',
            sft_recipe='Original full 5418-example vqa_knowledge_case_context',scopes=['attention','attention_ffn'],
            learning_rate=5e-5,seed=42,epochs=1,checkpoints='0 and saved checkpoints plus final; also disable fresh adapter at inference',
            resource='Run after acquisition experiment; original full SFT requires GPU queue availability.'),
        limitations=['Controlled acquisition of a small fact set, not a representative medical accuracy benchmark.',
            'Heldout question wording concerns exposed facts; four unexposed controls do not establish general medical generalization.',
            'Prose and QA differ in masked conditioning prefix and compute length, intentionally; supervised target tokens and exposures are matched.',
            'Two seeds are replications, not a model-size comparison. No claim of a 2B capacity ceiling without a size/control experiment.',
            'Some test phrasing is close to training; report both heldout questions separately and do not tune after seeing results.',
            'Conditional candidate likelihood is a retrieval probe; separately inspect generated content.']),
        )
    save(ROOT/'status.json',dict(state='prepared'))


def check_frozen():
    plan=read(ROOT/'plan.json')
    for path,digest in plan['frozen_files'].items():
        assert sha(Path(path))==digest,path
    return plan


def tokens_for(row, scope, exposure, eos):
    prefix=[eos] if scope=='prose' else row['question_prefix_ids'][0 if scope=='single_question' else exposure%3]
    return prefix+row['supervised_ids'],[-100]*len(prefix)+row['supervised_ids']


def load_base():
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig,Qwen3_5ForConditionalGeneration
    directory=LAB/'models/Qwen3.5-2B'
    # pread avoids the Windows mmap crash without duplicating the entire file in RAM.
    with safe_open(next(directory.glob('*.safetensors')),framework='pt',device='cpu',backend='pread') as weights:
        return Qwen3_5ForConditionalGeneration.from_pretrained(None,
            config=AutoConfig.from_pretrained(directory,local_files_only=True),
            state_dict={key:weights.get_tensor(key) for key in weights.keys()},
            dtype=torch.bfloat16,attn_implementation='sdpa')


def available_commit_mib():
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[
            (name,ctypes.c_ulonglong) for name in ['total_phys','avail_phys','total_page_file',
                'avail_page_file','total_virtual','avail_virtual','avail_extended_virtual']]
    status=MemoryStatus()
    status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.avail_page_file//1024**2


def schedule(rows, seed, steps):
    rng=random.Random(seed)
    stream=[]
    while len(stream)<steps*4:
        order=list(range(len(rows)))
        rng.shuffle(order)
        stream.extend(order)
    return stream[:steps*4]


def score_candidates(model, prefix, candidates):
    import torch
    result=[]
    for tokens in candidates:
        full=torch.tensor([prefix+tokens],device=model.device)
        with torch.inference_mode():
            logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False,
                logits_to_keep=len(tokens)+1).logits[0,:-1].float()
            losses=torch.nn.functional.cross_entropy(logits,torch.tensor(tokens,device=model.device),reduction='none')
        assert torch.isfinite(losses).all()
        result.append(dict(sum_nll=float(losses.sum()),mean_nll=float(losses.mean()),tokens=len(tokens)))
    return result


def evaluate(model,tok,out,step,rows):
    import torch
    path=out/'evaluation'/f'{step:06d}.json'
    if path.exists():
        return
    model.eval()
    partial=path.with_suffix('.jsonl')
    values=[json.loads(line) for line in partial.read_text(encoding='utf-8').splitlines()] if partial.exists() else []
    finished={(r['fact_id'],r['condition']) for r in values}
    for row in rows:
        answers=[row['answer']]+row['distractors']
        conditions=[('raw',row['prefix'],row['raw_prefix_ids']),
                    ('trained_question',row['training_questions'][0],prefix_ids(tok,row['training_questions'][0]))]
        conditions += [(f'heldout_{i}',q,prefix_ids(tok,q)) for i,q in enumerate(row['heldout_questions'])]
        for condition,prompt,prefix in conditions:
            if (row['id'],condition) in finished:continue
            save(ROOT/'status.json',dict(state='evaluating',run=out.name,step=step,
                completed=len(values),total=len(rows)*4,worker_pid=os.getpid()))
            is_raw=condition=='raw'
            if is_raw:
                prefix=[tok.eos_token_id]+prefix
            candidates=[tok.encode((' ' if is_raw else '')+a+'.',add_special_tokens=False) for a in answers]
            nll=score_candidates(model,prefix,candidates)
            sentence_nll=None if is_raw else score_candidates(model,prefix,
                [tok.encode(row['prefix']+' '+a+'.',add_special_tokens=False) for a in answers])
            margins={key:min(v[key] for v in nll[1:])-nll[0][key] for key in ['sum_nll','mean_nll']}
            inputs=torch.tensor([prefix],device=model.device)
            with torch.inference_mode():
                generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,
                    max_new_tokens=48,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)[0,len(prefix):].tolist()
            final=tok.decode(generated,skip_special_tokens=True).strip()
            record=dict(fact_id=row['id'],partition=row['partition'],condition=condition,
                answer=row['answer'],candidates=answers,nll=nll,gold_margin=margins,
                sentence_nll=sentence_nll,
                sentence_gold_margin=None if is_raw else min(v['sum_nll'] for v in sentence_nll[1:])-sentence_nll[0]['sum_nll'],
                prediction=final,generated_tokens=len(generated),stopped_on_eos=tok.eos_token_id in generated)
            values.append(record)
            partial.parent.mkdir(parents=True,exist_ok=True)
            with partial.open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
            if len(values)%16==0:print('EVAL_PROGRESS',out.name,step,len(values),flush=True)
    save(path,dict(step=step,records=values))
    summary={}
    for partition in ['trained','unexposed_control']:
        summary[partition]={}
        for condition in ['raw','trained_question','heldout_0','heldout_1']:
            group=[r for r in values if r['partition']==partition and r['condition']==condition]
            summary[partition][condition]=dict(n=len(group),**{key:sum(r['gold_margin'][key]>0 for r in group) for key in ['sum_nll','mean_nll']})
    save(out/'evaluation'/f'{step:06d}_summary.json',summary)
    print('EVAL',out.name,step,json.dumps(summary),flush=True)


def train(scope,seed,smoke=False):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoTokenizer,set_seed
    plan=check_frozen()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'])
    set_seed(seed)
    tok=AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B',local_files_only=True)
    all_rows=read(DATA/'facts.json');rows=[r for r in all_rows if r['partition']=='trained']
    out=ROOT/('smoke' if smoke else f'{scope}_seed{seed}')
    if (out/'complete.json').exists():
        return
    checkpoints=[2] if smoke else STEPS
    stream=schedule(rows,seed,checkpoints[-1])
    model=load_base()
    model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules=TARGETS,bias='none',task_type='CAUSAL_LM')).to('cuda')
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    assert any('.mlp.' in n and p.requires_grad for n,p in model.named_parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.config.use_cache=False
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],weight_decay=.01)
    previous=sorted((out/'checkpoints').glob('*/commit.json'))
    start=0
    if previous:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        p=previous[-1].parent
        commit=read(p/'commit.json')
        assert sha(p/'adapter/adapter_model.safetensors')==commit['adapter_sha256']
        assert sha(p/'state.pt')==commit['state_sha256']
        set_peft_model_state_dict(model,load_file(str(p/'adapter/adapter_model.safetensors')))
        state=torch.load(p/'state.pt',map_location='cpu',weights_only=False)
        optim.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        start=state['step']
    exposures=[0]*len(rows)
    for index in stream[:start*4]:
        exposures[index]+=1
    if not smoke and not start:
        evaluate(model,tok,out,0,all_rows)
    elif not smoke:
        evaluate(model,tok,out,start,all_rows)
    before=[p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n] if smoke else []
    torch.cuda.reset_peak_memory_stats()
    for step in range(start+1,checkpoints[-1]+1):
        model.train();optim.zero_grad(set_to_none=True)
        losses=[];target_tokens=0
        for index in stream[(step-1)*4:step*4]:
            ids,labels=tokens_for(rows[index],scope,exposures[index],tok.eos_token_id)
            exposures[index]+=1
            batch=torch.tensor([ids],device='cuda');targets=torch.tensor([labels],device='cuda')
            output=model(input_ids=batch,attention_mask=torch.ones_like(batch),labels=targets,use_cache=False)
            loss=output.loss
            assert torch.isfinite(loss),step
            (loss/4).backward();losses.append(float(loss.detach()));target_tokens+=sum(x!=-100 for x in labels)
            del output,loss,batch,targets
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm)
        optim.step()
        out.mkdir(parents=True,exist_ok=True)
        log=dict(step=step,loss=sum(losses)/4,grad_norm=float(norm),supervised_tokens=target_tokens,
            peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
        with (out/'train.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log)+'\n')
        save(ROOT/'status.json',dict(state='training',scope=scope,seed=seed,step=step,total=checkpoints[-1],worker_pid=os.getpid(),smoke=smoke))
        if step%12==0 or smoke:print(scope,seed,log,flush=True)
        if step in checkpoints:
            p=out/'checkpoints'/f'{step:06d}'
            model.save_pretrained(p/'adapter')
            torch.save(dict(step=step,optimizer=optim.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),p/'state.pt')
            save(p/'commit.json',dict(step=step,adapter_sha256=sha(p/'adapter/adapter_model.safetensors'),state_sha256=sha(p/'state.pt')))
            if not smoke:evaluate(model,tok,out,step,all_rows)
    if smoke:
        after=[p for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n]
        assert any(not torch.equal(a,b) for a,b in zip(before,after,strict=True))
    save(out/'complete.json',dict(steps=checkpoints[-1],exposures=exposures,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        adapter_sha256=sha(p/'adapter/adapter_model.safetensors'),smoke=smoke))
    del model,optim;gc.collect();torch.cuda.empty_cache()


def pipeline():
    prepare()
    for seed in SEEDS:
        for scope in SCOPES:
            out=ROOT/f'{scope}_seed{seed}'
            if (out/'complete.json').exists():continue
            # A short-sequence LoRA probe can share the GPU only with ample headroom.
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                commit=available_commit_mib()
                if free>=7500 and commit>=8000:break
                save(ROOT/'status.json',dict(state='waiting_for_memory',free_mib=free,required_mib=7500,
                    commit_free_mib=commit,commit_required_mib=8000,worker_pid=os.getpid()))
                time.sleep(60)
            subprocess.run([sys.executable,'-u',__file__,'train','--scope',scope,'--seed',str(seed)],cwd=LAB,check=True)
    check_frozen()
    save(ROOT/'status.json',dict(state='acquisition_complete',retention_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','smoke','train','pipeline'])
    parser.add_argument('--scope',choices=SCOPES,default='prose')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    prepare()
    try:
        if args.stage=='pipeline':pipeline()
        elif args.stage in ['smoke','train']:train(args.scope,args.seed,smoke=args.stage=='smoke')
    except Exception as error:
        save(ROOT/'status.json',dict(state='failed',error=repr(error),worker_pid=os.getpid()))
        raise
