"""Two isolated processes at a time, balanced arm order, immutable completed runs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'evals/results/architecture_compare_20260908_v2'
SCRIPT=PROJECT/'evals/architecture_compare_20260908.py'
env={**os.environ,'PYTHONIOENCODING':'utf-8'}


def run(mode,case,arm,rep):
    runid=f'{case}_{arm}_r{rep}'
    folder='scheduler' if mode=='scheduler' else 'runs'
    existing=ROOT/folder/runid/'result.json'
    if existing.exists():
        if json.loads(existing.read_text(encoding='utf-8'))['status']!='completed':
            raise RuntimeError(f'Inspect failed run before proceeding: {runid}')
        return
    proc=subprocess.run([sys.executable,str(SCRIPT),mode,'--case',case,'--arm',arm,'--rep',str(rep)],
                        cwd=PROJECT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=900)
    log=ROOT/'process_logs'/f'{runid}.txt'
    log.parent.mkdir(exist_ok=True)
    log.write_text(proc.stdout+proc.stderr,encoding='utf-8')
    print(proc.stdout.strip(),flush=True)
    if proc.returncode:
        print(proc.stderr,flush=True)
        raise RuntimeError(f'Execution failed: {runid}')


def pair(job):
    case,rep=job
    arms=['single','swarm'] if (int(case[1:])+rep)%2==0 else ['swarm','single']
    for arm in arms: run('run',case,arm,rep)


def scheduler(rep):
    arms=['serial','swarm'] if rep%2 else ['swarm','serial']
    for arm in arms: run('scheduler','S01',arm,rep)


def grade(path):
    runid=path.parent.name
    if (ROOT/'grades'/runid/'result.json').exists(): return
    proc=subprocess.run([sys.executable,str(SCRIPT),'grade','--runid',runid],cwd=PROJECT,env=env,
                        capture_output=True,text=True,encoding='utf-8',timeout=180)
    (ROOT/'process_logs'/f'grade_{runid}.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    print(proc.stdout.strip(),flush=True)
    if proc.returncode:
        print(proc.stderr,flush=True)
        raise RuntimeError(f'Grading failed: {runid}')


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(pair,[(f'A{i:02}',rep) for rep in (1,2) for i in range(1,9)]))
    # Scheduler timing is run alone, without competing patient cases or grading.
    for rep in (1,2,3): scheduler(rep)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(grade,sorted((ROOT/'runs').glob('*/result.json'))))
    print('ALL_RUNS_AND_GRADES_COMPLETED',flush=True)
