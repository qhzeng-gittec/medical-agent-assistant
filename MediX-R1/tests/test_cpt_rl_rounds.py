import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import training.train_cpt_rl_rounds as rl
import evaluation.rl_codex_judge as judging
import experiments.run_cpt_coverage_experiment as experiment


def raw_judgment(ids):
    return dict(requested_model=judging.MODEL, finish_reason='stop',
        usage=dict(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        message=dict(content=json.dumps(dict(reference_valid=True, candidates=[
            dict(id=i, answer_score=2, reasoning_score=1, flags=[], findings=[]) for i in ids]))))


class RlTests(unittest.TestCase):
    def test_group_advantages_and_invalid_reference(self):
        responses=[dict(format_score=1)]*4
        judgments=[dict(joint_score=s, reference_valid=True) for s in [0,1,2,2]]
        rewards, advantages=rl.group_advantages(judgments,responses)
        self.assertAlmostEqual(sum(advantages),0)
        self.assertLess(advantages[0],0)
        self.assertGreater(advantages[-1],0)
        self.assertEqual(rewards[-1],1)
        for j in judgments:
            j['reference_valid']=False
        self.assertEqual(rl.group_advantages(judgments,responses),([0]*4,[0]*4))
        self.assertEqual(rl.group_advantages([judgments[0]]*4,responses)[1],[0]*4)

    def test_judge_validation_and_cached_mapping(self):
        with self.assertRaises(ValueError):
            judging.parse_judgment(raw_judgment(['wrong']),['wanted'])
        raw=raw_judgment(['a'])
        result=json.loads(raw['message']['content'])
        result['candidates'][0]['answer_score']=True
        raw['message']['content']=json.dumps(result)
        with self.assertRaises(ValueError):
            judging.parse_judgment(raw,['a'])
        with tempfile.TemporaryDirectory() as directory:
            judge=judging.CodexJudge(directory,1)
            row=dict(task='knowledge',user_text='Question',reference='Reference',source_id='one')
            responses=[dict(reasoning='Evidence',final_answer='Answer'),dict(reasoning='Other',final_answer='Other')]
            def run(request, output_path):
                return raw_judgment([c['id'] for c in request['payload']['candidates']])
            with patch.object(judging,'judge_request',side_effect=run) as api:
                first=judge.score(row,responses)
                second=judge.score(row,list(reversed(responses)))
                self.assertEqual(first,list(reversed(second)))
                api.assert_called_once()
                with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                    judge.score(dict(row,user_text='Another question'),responses)
            self.assertEqual(judge.usage()['total_tokens'],130)

    def test_uncertain_attempt_never_automatically_billed_again(self):
        with tempfile.TemporaryDirectory() as directory:
            judge=judging.CodexJudge(directory,2)
            row=dict(task='case',user_text='Question',reference='Reference',source_id='one')
            responses=[dict(reasoning='Evidence',final_answer='Answer')]
            with patch.object(judging,'judge_request',side_effect=TimeoutError('uncertain')) as api:
                with self.assertRaises(TimeoutError):
                    judge.score(row,responses)
                with self.assertRaisesRegex(RuntimeError,'Unresolved prior'):
                    judge.score(row,responses)
                api.assert_called_once()

    def test_audited_truncation_replacement_is_cached_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            judge=judging.CodexJudge(Path(directory),2); root=judge.root
            row=dict(task='knowledge',user_text='Question',reference='Reference',source_id='one')
            responses=[dict(reasoning='Evidence',final_answer='Answer')]
            candidate=judging.digest(responses[0])[:20]
            truncated=raw_judgment([candidate]); truncated['finish_reason']='length'
            with patch.object(judging,'judge_request',return_value=truncated) as api:
                with self.assertRaises(ValueError):
                    judge.score(row,responses)
                original=next((root/'calls').glob('*.json'))
                request=json.loads((root/'requests'/original.name).read_text(encoding='utf-8'))
                replacement='a'*64
                (root/'recoveries').mkdir()
                judging.save(root/'recoveries'/original.name,dict(replacement_call_sha256=replacement))
                judging.save(root/'requests'/f'{replacement}.json',request)
                judging.save(root/'calls'/f'{replacement}.json',raw_judgment([candidate]))
                result=judge.score(row,responses)
                self.assertEqual(result[0]['call_sha256'],replacement)
                self.assertEqual(json.loads(original.read_text(encoding='utf-8'))['finish_reason'],'length')
                api.assert_called_once()
                request['thinking_level']='low'
                judging.save(root/'requests'/f'{replacement}.json',request)
                with self.assertRaisesRegex(ValueError,'differs'):
                    judge.score(row,responses)

    def test_frozen_128_plan_and_split(self):
        if not (rl.ROOT/'plan.json').exists():
            self.skipTest('Requires local frozen RL training artifacts; unit tests use synthetic fixtures.')
        plan=rl.read(rl.ROOT/'plan.json')
        self.assertEqual((plan['rounds'],plan['prompts_per_round'],plan['expected_steps']),(5,128,320))
        self.assertEqual(plan['planned_judge_calls'],1000)
        train=rl.load_jsonl(rl.ROOT/'train.jsonl')
        validation=rl.load_jsonl(rl.ROOT/'validation.jsonl')
        self.assertEqual((len(train),len(validation)),(128,60))
        self.assertFalse({r['source_id'] for r in train}&{r['source_id'] for r in validation})
        self.assertEqual(sum(r['task']=='knowledge' for r in train),96)

    def test_pipeline_inserts_rl_after_sft(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment,'ROOT',Path(directory)), patch.object(experiment,'EXPLORATORY',True):
            (Path(directory)/'plan.json').write_text(json.dumps(dict(rl={})))
            jobs=experiment.training_jobs()
            self.assertEqual([j[0] for j in jobs],['cpt','sft','rl','sft_only'])
            for job in [jobs[1],jobs[3]]:
                self.assertEqual(job[1][job[1].index('--lora-scope')+1],'attention')

    def test_adopt_sft_preserves_existing_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); adapter=root/'sft_seed42/recipe/final_adapter'
            adapter.mkdir(parents=True)
            for name in ['adapter_model.safetensors','adapter_config.json']:
                (adapter/name).touch()
            (adapter.parent/'run_config.json').write_text(json.dumps(dict(estimated_steps=1355)))
            (adapter.parent/'training_complete.json').write_text(json.dumps(dict(global_step=1355)))
            process=MagicMock()
            process.cmdline.return_value=['python','-m','training.sft','--output-root',str(root/'sft_seed42')]
            with patch.object(experiment,'ROOT',root), patch.object(experiment,'SFT',adapter), patch.object(experiment.psutil,'Process',return_value=process), patch.object(experiment.subprocess,'run') as launch:
                experiment.adopt_training(42,'sft')
                process.wait.assert_called_once()
                launch.assert_not_called()
                self.assertEqual(rl.read(root/'status.json')['next_stage'],'rl')

    def test_hybrid_qwen_rl_gradients_reference_and_optimizer_resume(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from peft import get_peft_model, LoraConfig
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config=Qwen3_5Config(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_key_head_dim=16,linear_value_head_dim=16,
            linear_num_key_heads=2,linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],
            max_position_embeddings=128,rope_parameters=dict(rope_type='default',rope_theta=10000.,
            partial_rotary_factor=.5,mrope_section=[1,1,2])),vision_config=dict(depth=1,hidden_size=32,
            intermediate_size=64,num_heads=2,out_hidden_size=32,num_position_embeddings=16),tie_word_embeddings=True)
        model=get_peft_model(Qwen3_5ForConditionalGeneration(config),LoraConfig(r=2,lora_alpha=4,
            target_modules=rl.TARGETS,lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
        inputs=dict(input_ids=torch.tensor([[2,3,4]]),attention_mask=torch.ones(1,3,dtype=torch.long))
        reply=[5,6,7]
        model.eval()
        with torch.no_grad():
            old=rl.response_logprobs(model,inputs,reply)
            with model.disable_adapter():
                reference=rl.response_logprobs(model,inputs,reply)
            full=model(input_ids=torch.tensor([[2,3,4,5,6,7]]),use_cache=False).logits
            expected=torch.log_softmax(full[0,2:5].float(),-1).gather(1,torch.tensor(reply)[:,None]).squeeze(1)
        torch.testing.assert_close(old,expected)
        torch.testing.assert_close(old,reference)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.)
        new=rl.response_logprobs(model,inputs,reply)
        loss,ratio=rl.policy_loss(new,old,1.)
        loss.backward()
        self.assertEqual(float(ratio.detach()),1.)
        names=[n for n,p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
        self.assertTrue(any('.mlp.' in n for n in names))
        self.assertTrue(any('attn.' in n for n in names))
        self.assertTrue(all('lora_' in n for n in names))
        optimizer.step()
        with torch.no_grad(),model.disable_adapter():
            torch.testing.assert_close(rl.response_logprobs(model,inputs,reply),reference)
        resumed=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.)
        resumed.load_state_dict(optimizer.state_dict())
        self.assertEqual({int(s['step']) for s in resumed.state.values()},{1})


if __name__=='__main__':
    unittest.main()
