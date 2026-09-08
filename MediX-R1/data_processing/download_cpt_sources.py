"""Download pinned public medical text corpora for local coverage mining."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / 'data/cpt_sources_v1'
REPOS = {'textbooks': 'MedRAG/textbooks', 'pubmed': 'MedRAG/pubmed'}
STATPEARLS = 'https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def download(item):
    path = ROOT / item['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if receipt.exists() and path.exists() and path.stat().st_size == item['bytes']:
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        if previous['url'] == item['url']:
            return previous
    partial = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < item['bytes']:
                with requests.get(item['url'], headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'} if offset else {'Accept-Encoding': 'identity'}, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected download range')
                    with partial.open('ab' if append else 'wb') as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
            if partial.stat().st_size != item['bytes']:
                raise ValueError(f"Download size mismatch: {item['path']}")
            with partial.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            if item.get('sha256') and digest != item['sha256']:
                raise ValueError(f"Download checksum mismatch: {item['path']}")
            partial.replace(path)
            result = dict(item, actual_sha256=digest, completed_at=time.time())
            save(receipt, result)
            return result
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['textbooks', 'statpearls', 'pubmed'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / f'{args.dataset}_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        if args.dataset == 'statpearls':
            response = requests.head(STATPEARLS, timeout=30)
            response.raise_for_status()
            plan = dict(dataset='NCBI Bookshelf StatPearls', source=STATPEARLS,
                        files=[dict(path='statpearls/statpearls_NBK430685.tar.gz', url=STATPEARLS,
                                    bytes=int(response.headers['Content-Length']))])
        else:
            repo = REPOS[args.dataset]
            response = requests.get(f'https://huggingface.co/api/datasets/{repo}', params={'blobs': 'true'}, timeout=60)
            response.raise_for_status()
            metadata = response.json()
            plan = dict(dataset=repo, revision=metadata['sha'], files=[])
            for item in metadata['siblings']:
                name = item['rfilename']
                if name.startswith('chunk/') and name.endswith('.jsonl') or name == 'README.md':
                    plan['files'].append(dict(path=f'{args.dataset}/{name}', bytes=item['size'],
                                             sha256=item.get('lfs', {}).get('sha256'),
                                             url=f"https://huggingface.co/datasets/{repo}/resolve/{metadata['sha']}/{name}"))
        plan['total_bytes'] = sum(f['bytes'] for f in plan['files'])
        save(plan_path, plan)
    remaining = sum(f['bytes'] for f in plan['files'] if not (ROOT/f['path']).exists())
    if shutil.disk_usage(ROOT).free < remaining + 20 * 1024**3:
        raise RuntimeError('Insufficient space for corpus and 20 GiB working reserve')
    status_path = ROOT / f'{args.dataset}_status.json'
    status = dict(dataset=args.dataset, state='downloading', completed_files=0, completed_bytes=0,
                  total_files=len(plan['files']), total_bytes=plan['total_bytes'])
    save(status_path, status)
    print(json.dumps(status), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item): item for item in plan['files']}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                status.update(state='failed', failed_path=futures[future]['path'], error_type=type(error).__name__)
                save(status_path, status)
                for pending in futures:
                    pending.cancel()
                raise
            status['completed_files'] += 1
            status['completed_bytes'] += result['bytes']
            save(status_path, status)
            if status['completed_files'] % 20 == 0 or args.dataset != 'pubmed':
                print(json.dumps(status), flush=True)
    status['state'] = 'complete'
    save(status_path, status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
