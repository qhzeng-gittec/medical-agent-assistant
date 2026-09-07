import json
import shutil
import subprocess
from pathlib import Path


def call_teacher(prompt: str, images: list[Path], schema: dict, output_path: Path, model: str) -> dict:
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("The authenticated Codex CLI is required for annotation and judging.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = output_path.with_suffix(".schema.json")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    command = [
        codex, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
        "--model", model, "--sandbox", "read-only", "--config", 'model_reasoning_effort="low"',
        "--output-schema", str(schema_path.resolve()), "--output-last-message", str(output_path.resolve()),
    ]
    for path in images:
        command.extend(["--image", str(path.resolve())])
    command.append("-")
    result = subprocess.run(command, input=prompt, text=True, encoding="utf-8", capture_output=True, timeout=900)
    output_path.with_suffix(".log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Teacher exited {result.returncode}; see {output_path.with_suffix('.log')}")
    if not output_path.exists():
        raise RuntimeError(f"Teacher did not produce structured output: {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))
