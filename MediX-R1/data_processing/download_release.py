"""Download the public dataset and materialize the original training layout."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from common.io import save_jsonl


def materialize(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a dataset directory: {output}")
    output.mkdir(parents=True)
    metadata = json.loads((source / 'release_manifest.json').read_text(encoding='utf-8'))
    records = {}
    counts = {}
    for split in ('train', 'validation', 'test'):
        rows = []
        parquet = source / f'data/{split}.parquet'
        with parquet.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != metadata['splits'][split]['parquet_sha256']:
            raise ValueError(f'Parquet checksum mismatch: {split}')
        for item in pq.read_table(parquet).to_pylist():
            row = json.loads(item['record_json'])
            if item['image']:
                path = output / row['image_path']
                if not path.resolve().is_relative_to(output.resolve()):
                    raise ValueError('Image path escapes dataset directory')
                path.parent.mkdir(parents=True, exist_ok=True)
                image_bytes = item['image']['bytes']
                if hashlib.sha256(image_bytes).hexdigest() != metadata['images'][row['image_path']]:
                    raise ValueError(f"Image checksum mismatch: {row['source_id']}")
                if not path.exists():
                    path.write_bytes(image_bytes)
            rows.append(row)
        if len(rows) != metadata['splits'][split]['rows']:
            raise ValueError(f'Row count mismatch: {split}')
        save_jsonl(output / f'{split}.jsonl', rows)
        counts[split] = len(rows)
        records[split] = {r['source_id']: r for r in rows}
    (output / 'manifest.json').write_text((source / 'source_manifest.json').read_text(encoding='utf-8'), encoding='utf-8')
    for name, recipe in metadata['recipes'].items():
        destination = output / 'recipes' / name
        destination.mkdir(parents=True)
        manifest = json.loads((source / 'recipes' / name / 'manifest.json').read_text(encoding='utf-8'))
        for split in ('train', 'validation'):
            rows = []
            for source_id in recipe[split]:
                row = dict(records[split][source_id])
                if row.get('image_path'):
                    row['image_path'] = os.path.relpath(output / row['image_path'], destination).replace('\\', '/')
                rows.append(row)
            target = destination / f'{split}.jsonl'
            save_jsonl(target, rows)
            manifest[f'original_{split}_sha256'] = manifest.get(f'{split}_sha256')
            manifest[f'{split}_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True, help='Dataset commit SHA or release tag')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/knowledge_experiments_v1')
    args = parser.parse_args()
    source = Path(snapshot_download(args.repo_id, repo_type='dataset', revision=args.revision))
    print(json.dumps(materialize(source, args.output)))


if __name__ == '__main__':
    main()
