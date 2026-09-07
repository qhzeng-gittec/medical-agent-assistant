import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
from filelock import FileLock


LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'outputs/knowledge_experiments_v1'
RUNS = [
    ('vqa', 'attention'),
    ('vqa', 'attention_ffn'),
    ('vqa_knowledge', 'attention'),
    ('vqa_knowledge', 'attention_ffn'),
    ('vqa_knowledge_case', 'attention_ffn'),
    ('vqa_knowledge_case_context', 'attention_ffn'),
]


def initial_state():
    jobs = []
    for recipe, scope in RUNS:
        jobs.append(dict(name=f'train_{recipe}_{scope}', kind='train', command=[sys.executable, '-u', "-m", "training.sft",
            '--recipes-dir', 'data/knowledge_experiments_v1/recipes', '--recipe', recipe,
            '--output-root', str(ROOT / 'seed42'), '--lora-scope', scope, '--batch-size', '2',
            '--gradient-accumulation-steps', '2', '--image-max-edge', '768', '--epochs', '1',
            '--learning-rate', '0.00005', '--eval-steps', '250', '--seed', '42']))
    for recipe, scope in [(None, None)] + RUNS:
        name = 'base' if recipe is None else f'{recipe}_{scope}'
        command = [sys.executable, '-u', "-m", "evaluation.evaluate_native_reasoning", '--name', name,
            '--data-dir', 'data/knowledge_experiments_v1', '--output-root', str(ROOT/'validation_screen'),
            '--split', 'validation', '--tasks', 'vqa', 'knowledge', 'case', 'context',
            '--limit-per-task', '20', '--image-max-edge', '768', '--decoding', 'greedy', '--seed', '42']
        if recipe is not None:
            command += ['--adapter-dir', str(ROOT/'seed42'/name/'final_adapter')]
        jobs.append(dict(name=f'evaluate_{name}', kind='validation_screen', command=command))
    return dict(status='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 phase='single_seed_initial_screen', seed=42, epochs=1, batch_size=2, gradient_accumulation_steps=2,
                 effective_batch_size=4, image_max_edge=768, visual_encoder='frozen',
                 evaluation='Same fixed 20 validation examples per task; exploratory screen, not full test or multi-seed evidence.',
                 comparison='Equal common-source exposure across additive recipes; extra data adds training steps. Attention/FFN pairs share exactly the same dataset.',
                 jobs=[dict(job,status='pending') for job in jobs])


def job_output(job):
    command = job['command']
    output_root = Path(command[command.index('--output-root') + 1])
    if job['kind'] == 'train':
        recipe = command[command.index('--recipe') + 1]
        scope = command[command.index('--lora-scope') + 1]
        return output_root / f'{recipe}_{scope}'
    return output_root / command[command.index('--name') + 1] / command[command.index('--split') + 1]


def latest_checkpoint(output):
    required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin')
    checkpoints = sorted(
        (p for p in output.glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]), reverse=True,
    )
    for checkpoint in checkpoints:
        if all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size for name in required):
            saved = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
            if saved['global_step'] != int(checkpoint.name[11:]):
                raise ValueError(f'Checkpoint step mismatch: {checkpoint}')
            return checkpoint
    return None


def prepare_command(job):
    command = list(job['command'])
    command[0] = sys.executable
    for flag in ('--resume-from-checkpoint', '--gpu-memory-fraction'):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    command += ['--gpu-memory-fraction', '0.8']
    if job['kind'] == 'train':
        checkpoint = latest_checkpoint(job_output(job))
        if checkpoint is not None:
            command += ['--resume-from-checkpoint', str(checkpoint)]
    return command


def archive_partial_output(job):
    output = job_output(job)
    partial = output / 'final_adapter' if job['kind'] == 'train' else output
    if not partial.exists():
        return
    archive = partial.with_name(partial.name + '.interrupted-' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    for path in (partial, archive):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Refusing to move output outside experiment root: {path}')
    partial.rename(archive)
    job.setdefault('preserved_partial_outputs', []).append(str(archive))


def save_state(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = ROOT / 'queue_status.json.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(ROOT / 'queue_status.json')


def assert_no_active_job():
    root = str(ROOT).replace('\\', '/').lower()
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if any(Path(arg).name in ('training.sft', 'evaluation.evaluate_native_reasoning') for arg in command):
            if root in ' '.join(command).replace('\\', '/').lower():
                raise RuntimeError(f'Experiment child is still running: PID {process.pid}')


def run_queue(state, dry_run=False):
    assert_no_active_job()
    state.update(status='running', pid=os.getpid(), gpu_memory_fraction=0.8)
    state.pop('interruption_reason', None)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    for index, job in enumerate(state['jobs']):
        if job['status'] == 'complete':
            continue
        marker = job_output(job) / ('training_complete.json' if job['kind'] == 'train' else 'summary.json')
        if marker.exists():
            json.loads(marker.read_text(encoding='utf-8'))
            job.update(status='complete', returncode=0, completion_recovered_from=str(marker))
            if not dry_run:
                save_state(state)
            continue
        command = prepare_command(job)
        if dry_run:
            print(json.dumps({'name': job['name'], 'command': command}, ensure_ascii=False))
            continue
        archive_partial_output(job)
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
        attempt = dict(started_at=datetime.now(timezone.utc).isoformat(), command=command,
                       log=str(logs / f"{job['name']}.{stamp}.log"))
        job.setdefault('attempts', []).append(attempt)
        job.update(status='running', started_at=attempt['started_at'], log=attempt['log'])
        job.pop('returncode', None)
        job.pop('finished_at', None)
        state['current_job'] = index
        save_state(state)
        with Path(job['log']).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=LAB, stdout=log, stderr=subprocess.STDOUT)
            attempt['pid'] = process.pid
            job['pid'] = process.pid
            save_state(state)
            returncode = process.wait()
        job.update(status='complete' if returncode == 0 else 'failed', returncode=returncode,
                   finished_at=datetime.now(timezone.utc).isoformat())
        attempt.update(returncode=returncode, finished_at=job['finished_at'])
        if returncode:
            state['status'] = 'failed'
            save_state(state)
            raise subprocess.CalledProcessError(returncode, command)
        save_state(state)
    if not dry_run:
        state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(ROOT / 'queue.lock', timeout=0):
        status_path = ROOT / 'queue_status.json'
        if args.resume:
            state = json.loads(status_path.read_text(encoding='utf-8-sig'))
        else:
            if status_path.exists():
                raise FileExistsError(f'Use --resume to continue existing queue: {status_path}')
            state = initial_state()
        run_queue(state, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
