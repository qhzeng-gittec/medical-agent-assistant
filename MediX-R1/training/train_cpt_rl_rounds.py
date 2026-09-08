"""Five on-policy medical GSPO rounds, Gemini rewards, and frozen validation monitoring."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if os.environ.get('MEDICAL_CPT_FAST_KERNELS') == '1':
    sys.path.insert(0, str(LAB/'.gpu-kernels'))
    os.environ.setdefault('CC', str(LAB/'.gpu-kernels/triton/runtime/tcc/tcc.exe'))
    os.environ.setdefault('CUDA_PATH', str(LAB/'.gpu-kernels/triton/backends/nvidia'))
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from common.io import load_jsonl, save_jsonl
from data_processing.download_cpt_sources import save
from common.native_reasoning import SYSTEM_PROMPT, parse_completion
from evaluation.rl_gemini_judge import GeminiJudge, MODEL, RUBRIC, digest
from training.cpt import TARGETS
from training.gspo import format_score, policy_loss, trim_completion

EXPERIMENT = LAB/'outputs/cpt_expanded_candidate_v1/experiment'
ROOT = EXPERIMENT/'rl_attention_ffn_v1'
DATA = LAB/'data/knowledge_experiments_v1'
CPT = EXPERIMENT/'cpt_seed42/final_adapter'
SFT = EXPERIMENT/'sft_seed42/vqa_knowledge_case_context_attention/final_adapter'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def select_rows(rows, knowledge, case, seed):
    rng = random.Random(seed)
    selected = []
    for task, count in [('knowledge', knowledge), ('case', case)]:
        pool = sorted((r for r in rows if r['task']==task and not r['image_path']), key=lambda r:r['source_id'])
        selected += rng.sample(pool, count)
    rng.shuffle(selected)
    return selected


def prepare(prompts=64):
    if (ROOT/'plan.json').exists():
        if read(ROOT/'plan.json')['prompts_per_round'] != prompts:
            raise ValueError('Existing RL plan has a different size.')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    train = select_rows(load_jsonl(DATA/'train.jsonl'), prompts*3//4, prompts//4, 20260908)
    validation = select_rows(load_jsonl(DATA/'validation.jsonl'), 40, 20, 20260909)
    assert not {r['source_id'] for r in train} & {r['source_id'] for r in validation}
    assert not {r['user_text'].strip() for r in train} & {r['user_text'].strip() for r in validation}
    save_jsonl(ROOT/'train.jsonl', train)
    save_jsonl(ROOT/'validation.jsonl', validation)
    save(ROOT/'plan.json', dict(rounds=5, prompts_per_round=prompts, group_size=4, prompts_per_cycle=8,
        prompts_per_update=2, updates_per_round=prompts//2, expected_steps=5*prompts//2,
        lora_scope='attention_ffn', rank=8, alpha=16, target_modules=TARGETS, learning_rate=1e-6,
        kl_coefficient=.02, clip_low=.0003, clip_high=.0004, seed=42, max_new_tokens=512,
        sampling=dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0),
        validation_decoding='greedy; same 512-token budget at round 0 and after every round; no forced tokens',
        optimizer='AdamW, no weight decay, clip gradient norm to 1; optimizer retained across rounds',
        initialization='Original base -> merge CPT -> merge attention-only SFT -> fresh attention+FFN RL LoRA',
        reference_policy='Frozen merged CPT+SFT, accessed by disabling only the new RL adapter',
        reward='.9 * min(Gemini answer_score, reasoning_score)/2 + .1 * complete native format',
        invalid_reference='Zero advantage for that training group; report invalid validation judgments in the fixed denominator',
        judge_model=MODEL, judge_rubric_sha256=digest(RUBRIC), judge_workers=2,
        max_judge_calls=5*prompts+6*60+1, planned_judge_calls=5*prompts+6*60,
        max_judge_output_tokens=4096, max_judge_request_characters=30000,
        source_train_sha256=sha(DATA/'train.jsonl'), source_validation_sha256=sha(DATA/'validation.jsonl'),
        train_sha256=sha(ROOT/'train.jsonl'), validation_sha256=sha(ROOT/'validation.jsonl'),
        cpt_manifest_sha256=read(EXPERIMENT/'plan.json')['corpus_manifest_sha256'],
        primary='Greedy strict answer correctness on 40 fixed validation knowledge questions',
        secondary=['20 validation cases','joint correctness','reasoning correctness','format/truncation/length',
                   'training reward and pass-at-4','zero-advantage groups','sampled KL to SFT','nonzero gradient updates'],
        limitations=['Semantic CPT coverage remains unverified.',
                     'Validation knowledge was intentionally exposed to CPT; validation questions are excluded from RL updates.',
                     'Gemini provides a proxy reward and metric, not clinician ground truth; reward hacking remains possible.',
                     'No no-CPT RL arm: improvement cannot be attributed uniquely to knowledge acquired in CPT.',
                     'Five rounds revisit the same training questions with fresh on-policy samples. No adaptive test-set selection.',
                     'This bounded pilot measures knowledge and cases, not image perception or research-context retention.']))
    save(ROOT/'status.json', dict(stage='waiting_for_sft', state='prepared', gpu_rl_started=False))


def group_advantages(judgments, responses):
    rewards = np.array([.9*j['joint_score']/2+.1*r['format_score'] if j['reference_valid'] else 0
                        for j,r in zip(judgments, responses, strict=True)], dtype=np.float64)
    return rewards.tolist(), ((rewards-rewards.mean())/(rewards.std()+1e-6)).tolist()


def metrics(responses, judgments):
    count = len(responses)
    return dict(samples=count,
        answer_accuracy=sum(j['reference_valid'] and j['answer_score']==2 for j in judgments)/count,
        reasoning_accuracy=sum(j['reference_valid'] and j['reasoning_score']==2 for j in judgments)/count,
        joint_accuracy=sum(j['reference_valid'] and j['joint_score']==2 for j in judgments)/count,
        unjudgeable=sum(not j['reference_valid'] for j in judgments),
        format_rate=sum(r['format_score'] for r in responses)/count,
        truncation_rate=sum(not r['stopped_on_eos'] for r in responses)/count,
        mean_generated_tokens=sum(r['generated_tokens'] for r in responses)/count)


def prompt_inputs(processor, row):
    return processor.apply_chat_template([
        dict(role='system', content=SYSTEM_PROMPT),
        dict(role='user', content=[dict(type='text', text=row['user_text'])])],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
        return_dict=True, return_tensors='pt')


def generate(model, processor, row, count, sample, budget):
    inputs = prompt_inputs(processor, row)
    eos = processor.tokenizer.eos_token_id
    model.eval()
    options = dict(do_sample=sample, max_new_tokens=budget, eos_token_id=eos,
                   pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
    if sample:
        options.update(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
    results = []
    for _ in range(count):
        with torch.no_grad():
            output = model.generate(**{k:v.to(model.device) for k,v in inputs.items()}, **options)
        ids = trim_completion(output[0, inputs['input_ids'].shape[1]:].tolist(), eos)
        stopped = ids[-1] == eos
        text = processor.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False).strip()
        results.append(dict(source_id=row['source_id'], task=row['task'], completion_ids=ids,
            raw_prediction=text, **parse_completion(text), generated_tokens=len(ids), stopped_on_eos=stopped,
            format_score=format_score(text, stopped)))
        del output
    return results


def response_logprobs(model, inputs, ids):
    reply = torch.tensor([ids], device=model.device)
    batch = {k:v.to(model.device) for k,v in inputs.items()}
    batch['input_ids'] = torch.cat([batch['input_ids'], reply], dim=1)
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    if 'mm_token_type_ids' in batch:
        batch['mm_token_type_ids'] = torch.cat([batch['mm_token_type_ids'], torch.zeros_like(reply)], dim=1)
    logits = model(**batch, use_cache=False, logits_to_keep=len(ids)+1).logits[0, :-1]
    return -F.cross_entropy(logits.float(), reply[0], reduction='none')


def write_metrics(round_index, results, judge):
    baseline = read(ROOT/'round_0_metrics.json') if round_index else results
    current = results['items']
    prior = {i['source_id']:i for i in baseline['items']}
    paired = [int(i['judge']['reference_valid'] and i['judge']['answer_score']==2)-
              int(prior[i['source_id']]['judge']['reference_valid'] and prior[i['source_id']]['judge']['answer_score']==2)
              for i in current if i['task']=='knowledge']
    rng = np.random.default_rng(42)
    ci = np.quantile(rng.choice(paired, size=(2000,len(paired))).mean(axis=1), [.025,.975])
    results['knowledge_delta_vs_sft'] = dict(delta=float(np.mean(paired)), bootstrap_ci95=ci.tolist())
    results['judge_usage'] = judge.usage()
    save(ROOT/f'round_{round_index}_metrics.json', results)
    history = [read(ROOT/f'round_{i}_metrics.json') for i in range(round_index+1)]
    with (ROOT/'metrics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['round','updates','knowledge_accuracy','case_accuracy','joint_accuracy',
            'format_rate','mean_tokens','training_reward','sampled_kl','zero_advantage_groups','nonzero_updates'])
        writer.writeheader()
        for h in history:
            t=h.get('training', {})
            writer.writerow(dict(round=h['round'],updates=h['updates'],knowledge_accuracy=h['knowledge']['answer_accuracy'],
                case_accuracy=h['case']['answer_accuracy'],joint_accuracy=h['overall']['joint_accuracy'],
                format_rate=h['overall']['format_rate'],mean_tokens=h['overall']['mean_generated_tokens'],
                training_reward=t.get('reward_mean'),sampled_kl=t.get('sampled_kl_mean'),
                zero_advantage_groups=t.get('zero_advantage_groups'),nonzero_updates=t.get('nonzero_updates')))


def evaluate_round(model, processor, rows, judge, round_index, updates, policy_hash, training):
    folder = ROOT/'validation'/f'round_{round_index}'
    folder.mkdir(parents=True, exist_ok=True)
    responses = []
    for row in rows:
        path = folder/f"{row['source_id']}.json"
        if path.exists():
            record = read(path)
            if record['policy_sha256'] != policy_hash:
                raise ValueError('Cached validation output belongs to a different checkpoint.')
            response = record['response']
        else:
            response = generate(model, processor, row, 1, False, 512)[0]
            save(path, dict(policy_sha256=policy_hash, response=response))
        responses.append(response)
    with ThreadPoolExecutor(max_workers=2) as pool:
        judgments = list(pool.map(lambda pair:judge.score(pair[0],[pair[1]])[0], zip(rows,responses)))
    result = dict(round=round_index,updates=updates,policy_sha256=policy_hash,training=training,
                  overall=metrics(responses,judgments),items=[dict(r,judge=j) for r,j in zip(responses,judgments)])
    for task in ['knowledge','case']:
        selected = [i for i,r in enumerate(rows) if r['task']==task]
        result[task] = metrics([responses[i] for i in selected],[judgments[i] for i in selected])
    write_metrics(round_index,result,judge)
    print(json.dumps(dict(round=round_index,updates=updates,knowledge=result['knowledge'],case=result['case'],
                          delta=result['knowledge_delta_vs_sft'],training=training),ensure_ascii=False),flush=True)


def load_policy(plan, checkpoint=None):
    torch.manual_seed(plan['seed'])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(LAB/'models/Qwen3.5-2B',
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    for adapter in [CPT,SFT]:
        base = PeftModel.from_pretrained(base,adapter).merge_and_unload(safe_merge=True)
        del base.peft_config
    base.config.use_cache = False
    base.enable_input_require_grads()
    if checkpoint:
        model = PeftModel.from_pretrained(base, checkpoint/'adapter', is_trainable=True)
    else:
        model = get_peft_model(base,LoraConfig(r=plan['rank'],lora_alpha=plan['alpha'],lora_dropout=0.,
            target_modules=plan['target_modules'],bias='none',task_type='CAUSAL_LM'))
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n and '.visual.' not in n for n in trainable)
    assert any('.mlp.' in n for n in trainable) and any('attn.' in n for n in trainable)
    save(ROOT/'trainable_parameters.json',dict(count=sum(p.numel() for p in trainable.values()),names=list(trainable)))
    return model.to('cuda'), list(trainable.values())


def run():
    plan = read(ROOT/'plan.json')
    if (ROOT/'training_complete.json').exists():
        return
    for name in ['train','validation']:
        assert sha(ROOT/f'{name}.jsonl') == plan[name+'_sha256']
        assert sha(DATA/f'{name}.jsonl') == plan['source_'+name+'_sha256']
    assert digest(RUBRIC)==plan['judge_rubric_sha256']
    assert read(CPT.parent/'run_config.json')['corpus_manifest_sha256']==plan['cpt_manifest_sha256']
    for adapter in [CPT,SFT]:
        complete=read(adapter.parent/'training_complete.json')
        assert complete['global_step']==complete['expected_steps'] and not complete.get('probe_only',False)
    assert read(SFT.parent/'run_config.json')['lora_scope']=='attention'
    lineage=dict(cpt_sha256=sha(CPT/'adapter_model.safetensors'),sft_sha256=sha(SFT/'adapter_model.safetensors'),
        rl_plan_sha256=sha(ROOT/'plan.json'),loading_order=plan['initialization'])
    assert read(SFT.parent/'run_config.json')['base_adapter_sha256']==lineage['cpt_sha256']
    if (ROOT/'lineage.json').exists():
        assert read(ROOT/'lineage.json')==lineage
    else:
        save(ROOT/'lineage.json',lineage)
    checkpoints=sorted((ROOT/'updates').glob('*/commit.json'))
    checkpoint=checkpoints[-1].parent if checkpoints else None
    if checkpoint:
        commit=read(checkpoint/'commit.json')
        assert commit['lineage']==lineage and sha(checkpoint/'adapter/adapter_model.safetensors')==commit['adapter_sha256']
        assert sha(checkpoint/'state.pt')==commit['state_sha256']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    model, parameters=load_policy(plan,checkpoint)
    processor=AutoProcessor.from_pretrained(LAB/'models/Qwen3.5-2B',local_files_only=True,do_resize=False)
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=0.)
    steps=[]
    start_cycle=0
    if checkpoint:
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        steps=state['steps']; start_cycle=state['next_cycle']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        assert {int(v['step']) for v in optimizer.state.values()} == {len(steps)}
    train=load_jsonl(ROOT/'train.jsonl'); validation=load_jsonl(ROOT/'validation.jsonl')
    judge=GeminiJudge(ROOT/'judge',plan['max_judge_calls'])
    if not (ROOT/'round_0_metrics.json').exists():
        assert not checkpoint
        save(ROOT/'status.json',dict(stage='rl_validation',state='running',round=0,updates=0,worker_pid=os.getpid()))
        evaluate_round(model,processor,validation,judge,0,0,digest(lineage),{})
    cycles_per_round=len(train)//plan['prompts_per_cycle']
    for round_index in range(1,plan['rounds']+1):
        order=list(train); random.Random(plan['seed']+round_index).shuffle(order)
        for position in range(cycles_per_round):
            cycle=(round_index-1)*cycles_per_round+position
            if cycle<start_cycle:
                continue
            folder=ROOT/'cycles'/f'{cycle:04d}'; folder.mkdir(parents=True,exist_ok=True)
            rows=order[position*8:(position+1)*8]
            save(ROOT/'status.json',dict(stage='rl_rollout',state='running',round=round_index,cycle=cycle,
                updates=len(steps),worker_pid=os.getpid(),gpu_rl_started=True))
            rollout_path=folder/'rollouts.json'
            if rollout_path.exists():
                groups=read(rollout_path)
                assert [g['source_id'] for g in groups]==[r['source_id'] for r in rows]
            else:
                torch.manual_seed(plan['seed']+cycle)
                groups=[dict(source_id=r['source_id'],responses=generate(model,processor,r,4,True,plan['max_new_tokens'])) for r in rows]
                save(rollout_path,groups)
            with ThreadPoolExecutor(max_workers=2) as pool:
                judgments=list(pool.map(lambda pair:judge.score(pair[0],pair[1]['responses']),zip(rows,groups)))
            batch=[]
            model.train()
            for row,group,judgment in zip(rows,groups,judgments,strict=True):
                rewards,advantages=group_advantages(judgment,group['responses'])
                inputs=prompt_inputs(processor,row)
                for response,j,reward,advantage in zip(group['responses'],judgment,rewards,advantages,strict=True):
                    with torch.no_grad():
                        old=response_logprobs(model,inputs,response['completion_ids']).detach().cpu()
                        with model.disable_adapter():
                            ref=response_logprobs(model,inputs,response['completion_ids']).detach().cpu()
                    assert torch.isfinite(old).all() and torch.isfinite(ref).all()
                    batch.append(dict(inputs=inputs,response=response,judge=j,reward=reward,advantage=advantage,old=old,ref=ref))
            save(folder/'judgments.json',judgments)
            for offset in range(0,len(batch),8):
                optimizer.zero_grad(set_to_none=True)
                minibatch=batch[offset:offset+8]
                kls=[]; losses=[]; ratios=[]
                for item in minibatch:
                    new=response_logprobs(model,item['inputs'],item['response']['completion_ids'])
                    loss,ratio=policy_loss(new,item['old'].to(model.device),item['advantage'])
                    difference=item['ref'].to(model.device)-new
                    kl=(difference.exp()-difference-1).mean()
                    objective=(loss+plan['kl_coefficient']*kl)/len(minibatch)
                    if not torch.isfinite(objective):
                        raise RuntimeError('Nonfinite RL objective.')
                    objective.backward()
                    kls.append(float(kl.detach())); losses.append(float(loss.detach())); ratios.append(float(ratio.detach()))
                    del new,loss,ratio,kl,objective,difference
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
                optimizer.step()
                steps.append(dict(step=len(steps)+1,round=round_index,cycle=cycle,grad_norm=norm,
                    policy_loss=float(np.mean(losses)),sampled_kl=float(np.mean(kls)),ratios=ratios,
                    reward_mean=float(np.mean([i['reward'] for i in minibatch])),
                    zero_advantage_groups=sum(all(i['advantage']==0 for i in minibatch[k:k+4]) for k in [0,4]),
                    pass_at_4_groups=sum(any(i['judge']['reference_valid'] and i['judge']['answer_score']==2 for i in minibatch[k:k+4]) for k in [0,4])))
                print(json.dumps(steps[-1]),flush=True)
            checkpoint=ROOT/'updates'/f'{len(steps):06d}'
            checkpoint.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(checkpoint/'adapter')
            torch.save(dict(optimizer=optimizer.state_dict(),steps=steps,next_cycle=cycle+1,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),checkpoint/'state.tmp')
            (checkpoint/'state.tmp').replace(checkpoint/'state.pt')
            save(checkpoint/'commit.json',dict(lineage=lineage,adapter_sha256=sha(checkpoint/'adapter/adapter_model.safetensors'),
                state_sha256=sha(checkpoint/'state.pt')))
            save(ROOT/'progress.json',dict(round=round_index,cycle=cycle,updates=len(steps),last_checkpoint=str(checkpoint),judge_usage=judge.usage()))
            del batch
            optimizer.zero_grad(set_to_none=True)
        round_checkpoint=ROOT/'updates'/f'{round_index*plan["updates_per_round"]:06d}'
        if (ROOT/f'round_{round_index}_metrics.json').exists():
            continue
        if len(steps)!=round_index*plan['updates_per_round']:
            raise ValueError('Missing earlier round evaluation; cannot evaluate it using a later policy.')
        selected=[s for s in steps if s['round']==round_index]
        training=dict(reward_mean=float(np.mean([s['reward_mean'] for s in selected])),
            sampled_kl_mean=float(np.mean([s['sampled_kl'] for s in selected])),
            zero_advantage_groups=sum(s['zero_advantage_groups'] for s in selected),
            nonzero_updates=sum(s['grad_norm']>0 for s in selected),
            training_pass_at_4=sum(s['pass_at_4_groups'] for s in selected)/len(train))
        save(ROOT/'status.json',dict(stage='rl_validation',state='running',round=round_index,updates=len(steps),worker_pid=os.getpid()))
        evaluate_round(model,processor,validation,judge,round_index,len(steps),sha(round_checkpoint/'adapter/adapter_model.safetensors'),training)
    assert len(steps)==plan['expected_steps']
    if not any(s['grad_norm']>0 for s in steps):
        raise RuntimeError('No nonzero RL gradients; this is not evidence of learning.')
    model.save_pretrained(ROOT/'final_adapter')
    processor.save_pretrained(ROOT/'final_adapter')
    save(ROOT/'final_adapter/base_initialization.json',lineage)
    save(ROOT/'training_complete.json',dict(global_step=len(steps),expected_steps=plan['expected_steps'],
        rounds=plan['rounds'],probe_only=False,nonzero_updates=sum(s['grad_norm']>0 for s in steps),
        judge_usage=judge.usage(),final_adapter_sha256=sha(ROOT/'final_adapter/adapter_model.safetensors')))
    save(ROOT/'status.json',dict(stage='rl',state='complete',rounds=plan['rounds'],updates=len(steps)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--prompts',type=int,choices=[64,128],default=64)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare(args.prompts)
    else:
        try:
            run()
        except Exception as error:
            save(ROOT/'status.json',dict(state='failed',error=str(error),worker_pid=os.getpid()))
            raise
