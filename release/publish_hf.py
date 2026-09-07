"""Upload the prepared, verified dataset and seven final adapters to Hugging Face."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, RemoteEntryNotFoundError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', required=True, type=Path)
    parser.add_argument('--namespace', required=True, help='Your Hugging Face username')
    parser.add_argument('--upload', action='store_true', help='Without this flag, only show the upload plan')
    args = parser.parse_args()
    registry = json.loads((args.packages / 'registry.json').read_text(encoding='utf-8'))
    plan = [('dataset', registry['dataset'], args.packages / 'dataset-medix-sft')]
    plan += [('model', model['repository'], args.packages / model['directory']) for model in registry['models'].values()]
    manifests = {}
    for kind, name, folder in plan:
        manifest = json.loads((folder / 'release_manifest.json').read_text(encoding='utf-8'))
        targets = {'adapter_model.safetensors': manifest['adapter_sha256']} if kind == 'model' else {
            f'data/{split}.parquet': details['parquet_sha256'] for split, details in manifest['splits'].items()}
        for filename, expected in targets.items():
            with (folder / filename).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Artifact checksum mismatch: {name}/{filename}')
        manifests[name] = manifest
    print(json.dumps([{'type': kind, 'repo_id': f'{args.namespace}/{name}', 'source': str(folder)} for kind, name, folder in plan], ensure_ascii=False, indent=2))
    if not args.upload:
        return
    if not get_token():
        raise RuntimeError('Sign in locally with hf auth login first. Do not put a token in source code.')
    api = HfApi()
    if api.whoami()['name'] != args.namespace:
        raise ValueError('The signed-in account differs from the requested namespace')
    for kind, name, folder in plan:
        repo_id = f'{args.namespace}/{name}'
        try:
            api.repo_info(repo_id, repo_type=kind)
        except RepositoryNotFoundError:
            api.create_repo(repo_id, repo_type=kind, private=False)
        else:
            try:
                remote = hf_hub_download(repo_id, 'release_manifest.json', repo_type=kind)
            except RemoteEntryNotFoundError as exc:
                raise RuntimeError(f'Refusing to overwrite an existing unrecognized repository: {repo_id}') from exc
            if json.loads(Path(remote).read_text(encoding='utf-8')) != manifests[name]:
                raise ValueError(f'Existing repository contains a different release: {repo_id}')
        result = api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=folder,
                                   allow_patterns=['*.json', '*.md', '*.safetensors', '*.jinja', 'LICENSE', 'licenses/*', 'data/*.parquet', 'recipes/*/manifest.json'],
                                   commit_message='Publish verified medical assistant research artifacts')
        print(f'{repo_id}: {result.oid}', flush=True)


if __name__ == '__main__':
    main()
