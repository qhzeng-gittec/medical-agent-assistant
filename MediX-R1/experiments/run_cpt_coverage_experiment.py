"""Paired SFT versus knowledge-covered CPT+SFT, gated on a frozen coverage audit."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

from common.io import load_jsonl
from data_processing.download_cpt_sources import save

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB/'outputs/cpt_expanded_coverage_v1/experiment'
DATA = LAB/'data/cpt_expanded_coverage_v1'
SOURCE = LAB/'data/knowledge_experiments_v1'
RECIPE = 'vqa_knowledge_case_context'
SFT_SCOPE = 'attention'
BASELINE = ROOT/f'sft_only_seed42/{RECIPE}_{SFT_SCOPE}/final_adapter'
CPT = ROOT/'cpt_seed42/final_adapter'
SFT = ROOT/f'sft_seed42/{RECIPE}_{SFT_SCOPE}/final_adapter'
EXPLORATORY = False


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():
        plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
        if plan.get('exploratory', False) != EXPLORATORY:
            raise ValueError('Existing experiment mode differs.')
        if plan['sft']['lora_scope'] != SFT_SCOPE:
            raise ValueError('Existing SFT scope differs.')
        return
    recipe_hash = sha(SOURCE/'recipes'/RECIPE/'train.jsonl')
    rows = load_jsonl(SOURCE/'test.jsonl')
    save(ROOT/'plan.json', dict(seed=42, arms=['sft_only','cpt_sft'], sft_recipe=RECIPE,
         exploratory=EXPLORATORY, semantic_full_coverage_verified=False,
         corpus_manifest_sha256=sha(DATA/'manifest.json') if EXPLORATORY else None,
         sft_train_sha256=recipe_hash, sft_validation_sha256=sha(SOURCE/'validation.jsonl'),
         evaluation_sha256=sha(SOURCE/'test.jsonl'), counts=dict(Counter(r['task'] for r in rows)),
         baseline_adapter_sha256=None, baseline_adapter_dir=str(BASELINE),
         baseline_reuse='Train a matching attention-only direct SFT baseline on the same four-task recipe.',
         corpus=str(DATA), required_coverage=6435, priority_retrieval_questions=1353,
         training_gate=('Explicitly authorized expanded-corpus experiment; semantic coverage unverified; token audit and frozen hashes required.' if EXPLORATORY else
                        'All required source IDs semantically reviewed; evidence passages reconstructed in actual CPT tokens; corpus hashes frozen.'),
         cpt=dict(rank=32, learning_rate=5e-5, epochs=1, max_length=2048, batch_size=1, gradient_accumulation_steps=4),
         sft=dict(lora_scope=SFT_SCOPE, rank=8, learning_rate=5e-5, epochs=1, batch_size=2, gradient_accumulation_steps=2, image_max_edge=768),
         generation=dict(decoding='greedy', enable_thinking=True, max_new_tokens=2048, reasoning_budget=1792),
         primary='Knowledge strict answer correctness on 150 frozen project test questions.',
         secondary=['Case answer correctness on 162 questions','context','vqa','paired wins and losses'],
         limitations=['Intentional exposure to train/validation/test answer knowledge: controlled knowledge acquisition, not unseen-knowledge generalization.',
                      'One seed; additional CPT data and compute are not compute-matched to direct SFT.',
                      'No original question-answer pairs in CPT.',
                      'Text coverage does not establish image-specific perception.',
                      'External model scoring is a separate budgeted stage; this runner does not call a paid judge.']))
    save(ROOT/'status.json', dict(stage='prepare', state='candidate_prepared' if EXPLORATORY else 'waiting_for_verified_corpus', gpu_training_started=False))


def verify():
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    if plan.get('exploratory', False) != EXPLORATORY:
        raise ValueError('Experiment mode differs from the frozen plan.')
    if plan['sft']['lora_scope'] != SFT_SCOPE or Path(plan['baseline_adapter_dir']) != BASELINE:
        raise ValueError('SFT scope or baseline path differs from the plan.')
    if 'rl' in plan and sha(ROOT/'rl_attention_ffn_v1/plan.json') != plan['rl']['plan_sha256']:
        raise ValueError('The frozen RL plan changed.')
    assert sha(SOURCE/'recipes'/RECIPE/'train.jsonl') == plan['sft_train_sha256']
    assert sha(SOURCE/'validation.jsonl') == plan['sft_validation_sha256']
    assert sha(SOURCE/'test.jsonl') == plan['evaluation_sha256']
    if plan['baseline_adapter_sha256'] is not None:
        assert sha(BASELINE/'adapter_model.safetensors') == plan['baseline_adapter_sha256']
    coverage_path = DATA/'coverage_status.json'
    if not coverage_path.exists():
        raise ValueError('The expanded corpus has not passed semantic and token coverage review.')
    coverage = json.loads(coverage_path.read_text(encoding='utf-8'))
    if EXPLORATORY:
        manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        audit = json.loads((DATA/'audit.json').read_text(encoding='utf-8'))
        if manifest.get('experiment_type') != 'expanded_candidate_not_full_coverage' or manifest.get('semantic_full_coverage_verified') is not False:
            raise ValueError('Exploratory mode requires an explicitly unverified candidate corpus.')
        if audit['status'] != 'passed' or sha(DATA/'manifest.json') != plan['corpus_manifest_sha256']:
            raise ValueError('Candidate token audit or frozen manifest mismatch.')
        for name in ['train', 'validation', 'documents', 'evaluation', 'candidate_map']:
            if sha(DATA/f'{name}.jsonl') != manifest[name+'_sha256']:
                raise ValueError(f'Candidate file changed: {name}')
    elif not coverage['ready_for_training'] or coverage['required'] != 6435 or coverage['accepted'] != 6435:
        raise ValueError('Full coverage of all 6,435 required items has not been verified.')
    if coverage['corpus_manifest_sha256'] != sha(DATA/'manifest.json'):
        raise ValueError('Coverage audit and corpus manifest differ.')
    if sha(DATA/'evaluation.jsonl') != plan['evaluation_sha256']:
        raise ValueError('Evaluation source changed during corpus preparation.')


def train():
    verify()
    jobs = training_jobs()
    if EXPLORATORY:
        jobs[0][1].append('--allow-unverified-coverage')
    for stage, command, output in jobs:
        if (output/'training_complete.json').exists():
            complete = json.loads((output/'training_complete.json').read_text(encoding='utf-8'))
            if complete.get('probe_only'):
                raise ValueError('A probe is not a completed experiment.')
            continue
        checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints:
            command += ['--resume-from-checkpoint', str(checkpoints[-1])]
        save(ROOT/'status.json', dict(stage=stage, state='process_launched', worker_pid=os.getpid(),
                                     exploratory=EXPLORATORY, semantic_full_coverage_verified=False))
        with (ROOT/f'{stage}.log').open('a', encoding='utf-8') as log:
            subprocess.run(command, cwd=LAB, env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           MEDICAL_CPT_FAST_KERNELS=os.environ.get('MEDICAL_CPT_FAST_KERNELS', '0') if stage in ['cpt', 'rl'] else '0'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    for adapter in [SFT, BASELINE]:
        config = json.loads((adapter.parent/'run_config.json').read_text(encoding='utf-8'))
        assert config['recipe_sha256'] == plan['sft_train_sha256']
        assert config['lora_scope'] == SFT_SCOPE
        assert all('.mlp.' not in name for name in config['selected_modules'])
        for key in ['learning_rate', 'epochs', 'batch_size', 'gradient_accumulation_steps', 'image_max_edge']:
            assert config[key] == plan['sft'][key], key
        assert config['seed'] == plan['seed']
    plan['baseline_adapter_sha256'] = sha(BASELINE/'adapter_model.safetensors')
    save(ROOT/'plan.json', plan)
    save(ROOT/'status.json', dict(stage='training', state='complete'))


def training_jobs():
    sft_command = [sys.executable, '-u', '-m', 'training.sft', '--recipe', RECIPE,
                   '--recipes-dir', str(SOURCE/'recipes'), '--lora-scope', SFT_SCOPE, '--batch-size', '2',
                   '--gradient-accumulation-steps', '2', '--image-max-edge', '768', '--epochs', '1',
                   '--learning-rate', '0.00005', '--eval-steps', '250', '--seed', '42', '--gpu-memory-fraction', '.8']
    jobs = [('cpt', [sys.executable, '-u', '-m', 'training.cpt', '--data-dir', str(DATA), '--output-dir', str(CPT.parent)], CPT.parent),
            ('sft', sft_command+['--output-root', str(ROOT/'sft_seed42'), '--base-adapter-dir', str(CPT)], SFT.parent),
            ('sft_only', sft_command+['--output-root', str(ROOT/'sft_only_seed42')], BASELINE.parent)]
    if EXPLORATORY and (ROOT/'plan.json').exists() and 'rl' in json.loads((ROOT/'plan.json').read_text(encoding='utf-8')):
        jobs.insert(2, ('rl', [sys.executable, '-u', '-m', 'training.train_cpt_rl_rounds', 'run'], ROOT/'rl_attention_ffn_v1'))
    return jobs


def adopt_training(pid, stage):
    process = psutil.Process(pid)
    command = process.cmdline()
    adapter = CPT if stage == 'cpt' else SFT
    script, argument, output = ('training.cpt','--output-dir',CPT.parent) if stage=='cpt' else (
        'training.sft','--output-root',ROOT/'sft_seed42')
    if script not in command or argument not in command or Path(command[command.index(argument)+1]) != output:
        raise ValueError('The process does not belong to this experiment stage.')
    save(ROOT/'status.json', dict(stage=stage, state='adopted_running_training', worker_pid=os.getpid(),
                                 trainer_pid=pid, next_stage='sft' if stage=='cpt' else 'rl', sft_lora_scope=SFT_SCOPE,
                                 exploratory=EXPLORATORY, semantic_full_coverage_verified=False))
    print(f'Waiting for existing {stage} process {pid}; the revised pipeline will continue afterward.', flush=True)
    process.wait()
    complete = json.loads((adapter.parent/'training_complete.json').read_text(encoding='utf-8'))
    config = json.loads((adapter.parent/'run_config.json').read_text(encoding='utf-8'))
    expected = config['expected_steps'] if stage=='cpt' else config['estimated_steps']
    if complete.get('probe_only') or complete['global_step'] != expected:
        raise ValueError('Adopted process did not complete its prescribed training.')
    if not (adapter/'adapter_model.safetensors').exists() or not (adapter/'adapter_config.json').exists():
        raise ValueError('Completed adapter is missing.')


def generate():
    verify()
    if json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))['baseline_adapter_sha256'] is None:
        raise ValueError('The matching attention-only baseline has not completed.')
    import experiments.run_cpt_experiment as paired
    paired.ROOT, paired.DATA = ROOT, DATA
    paired.BASELINE, paired.CPT, paired.SFT = BASELINE, CPT, SFT
    paired.generate(['sft_only','cpt_sft'])


def wait():
    prepare()
    save(ROOT/'status.json', dict(stage='coverage', state='queued_waiting_for_verified_corpus',
                                 worker_pid=os.getpid(), gpu_training_started=False))
    print('Queued: waiting for semantic and token coverage approval; no GPU allocated.', flush=True)
    while True:
        path = DATA/'coverage_status.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['ready_for_training']:
            break
        time.sleep(30)
    train()
    generate()
    save(ROOT/'status.json', dict(stage='evaluation', state='predictions_ready_for_budgeted_scoring'))


def all_stages():
    prepare()
    train()
    generate()
    save(ROOT/'status.json', dict(stage='evaluation', state='predictions_ready_for_budgeted_scoring',
                                 exploratory=EXPLORATORY, semantic_full_coverage_verified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','train','generate','wait','all'])
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--adopt-cpt-pid', type=int, help='Wait for an existing CPT process, then continue the revised pipeline.')
    parser.add_argument('--adopt-sft-pid', type=int, help='Wait for existing SFT without restarting it, then continue to RL.')
    args = parser.parse_args()
    EXPLORATORY = args.exploratory
    if EXPLORATORY:
        if args.stage == 'wait':
            parser.error('Exploratory mode uses all; wait is reserved for verified full coverage.')
        ROOT = LAB/'outputs/cpt_expanded_candidate_v1/experiment'
        DATA = LAB/'data/cpt_expanded_candidate_v1'
        CPT = ROOT/'cpt_seed42/final_adapter'
        SFT = ROOT/f'sft_seed42/{RECIPE}_{SFT_SCOPE}/final_adapter'
        BASELINE = ROOT/f'sft_only_seed42/{RECIPE}_{SFT_SCOPE}/final_adapter'
    try:
        if args.adopt_cpt_pid or args.adopt_sft_pid:
            if args.stage != 'all':
                parser.error('Adoption requires all')
            if args.adopt_cpt_pid and args.adopt_sft_pid:
                parser.error('Adopt only one active training process')
            prepare()
            verify()
            adopt_training(args.adopt_cpt_pid or args.adopt_sft_pid, 'cpt' if args.adopt_cpt_pid else 'sft')
        (all_stages if args.stage == 'all' else globals()[args.stage])()
    except Exception as error:
        save(ROOT/'status.json', dict(state='failed', error=str(error), worker_pid=os.getpid(), exploratory=EXPLORATORY))
        raise
