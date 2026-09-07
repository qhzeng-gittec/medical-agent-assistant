import json
from pathlib import Path

from common.io import load_jsonl
from experiments.run_knowledge_experiments import initial_state


def test_portable_image_path_is_resolved_from_dataset_not_working_directory(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    file = dataset / 'train.jsonl'
    file.write_text(json.dumps({'source_id': 'image-1', 'image_path': 'images/one.png'}) + '\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert load_jsonl(file)[0]['image_path'] == str((dataset / 'images/one.png').resolve())


def test_experiment_commands_use_importable_modules_after_reorganization():
    root = Path(__file__).resolve().parents[1]
    for job in initial_state()['jobs']:
        command = job['command']
        assert command[1:3] == ['-u', '-m']
        assert (root / (command[3].replace('.', '/') + '.py')).is_file()
