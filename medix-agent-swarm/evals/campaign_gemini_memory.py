"""Codex GPT-5.5 comparison on the medical Agent and frozen recall cases."""

import argparse
import asyncio
import json
import sys
import threading
from collections import Counter

from loguru import logger

import campaign_memory_ablation as ablation
from campaign_gateway import ApiAccessBlocked, BudgetExceeded, Gateway, ModelAdapter, dump, sha
from campaign_run import OUTPUT, PROJECT


ROOT = OUTPUT / "codex_memory_ablation_v1"
from codex_gateway import MODEL, request as codex_request


class CodexGateway:
    def __init__(self, root=ROOT, embedding_gateway=None, max_requests=50):
        self.root = OUTPUT
        self.output = root
        self.embedding_gateway = embedding_gateway
        self.requests = 0
        self.max_requests = max_requests
        self.lock = threading.Lock()

    def request(self, endpoint, payload, trace, role):
        if endpoint == 'embeddings':
            if self.embedding_gateway is None:
                raise RuntimeError('Remote Qwen embedding gateway is not configured')
            return self.embedding_gateway.request(endpoint, payload, trace, role)
        if endpoint != 'chat/completions' or payload['model'] != MODEL:
            raise ValueError('Request outside this comparison scope')
        with self.lock:
            if self.requests >= self.max_requests:
                raise BudgetExceeded('Codex request cap reached')
            self.requests += 1
        return codex_request(self.output, payload, trace, role)

    async def chat(self, model, messages, trace, role, tools=None, tool_choice='auto', max_tokens=2048):
        payload = dict(model=model, messages=messages, max_tokens=max_tokens, reasoning_effort='low')
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        return await asyncio.to_thread(self.request, 'chat/completions', payload, trace, role)


async def tool_probe(gateway):
    path = ROOT / "tool_probe.json"
    if path.exists():
        if not ablation.read(path)["passed"]:
            raise RuntimeError("Previous tool probe failed; inspect before rerunning")
        return
    trace = []
    result = {"passed": False, "model": MODEL}
    try:
        adapter = ModelAdapter(gateway, MODEL, trace, "probe")
        tools = [{"type": "function", "function": {"name": "echo", "description": "Echo a value",
                 "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}]
        messages = [{"role": "user", "content": "Call echo with value OK, then reply with its returned value."}]
        first = await adapter.chat_with_tools(messages, tools, "required")
        if len(first.tool_calls) != 1 or first.tool_calls[0].arguments != {"value": "OK"}:
            raise ValueError("Echo arguments differ from the probe contract")
        call = first.tool_calls[0]
        if call.name != "echo":
            raise ValueError("Unexpected function name")
        messages += [first.wire_message, adapter.create_tool_message(call.id, call.name, {"value": "OK"})]
        second = await adapter.chat_with_tools(messages)
        if second.finish_reason != "stop" or second.tool_calls or not second.content or "OK" not in second.content:
            raise ValueError("Echo round trip did not produce a completed answer")
        result.update(passed=True, answer=second.content)
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        dump(path, {**result, "trace": trace})
    print(json.dumps(result), flush=True)


def report():
    rows = [ablation.read(p) for p in sorted((ROOT / "runs").glob("*.json"))]
    summary = {"model": MODEL, "provider": "codex_chatgpt", "planned_runs": 16,
               "statuses": dict(Counter(r["status"] for r in rows)), "independent_source_cases": 4,
               "single_sample_per_arm": True, "cloud_recall": "frozen actual Mem0 r1 results",
               "cloud_write_back": False, "clinical_validation": False,
               "protocol_failure_count": sum(len(r["protocol"]["failures"]) for r in rows),
               "response_contract_failure_count": sum(len(r["protocol"]["response_contract_failures"]) for r in rows),
               "estimated_target_usd_not_invoice": sum(r["metrics"]["target_cost_usd"] for r in rows),
               "model_requests": sum(r["metrics"]["model_requests"] for r in rows),
               "retrieval_executions": sum(r["metrics"]["retrieval_backend_executions"] for r in rows)}
    dump(ROOT / "summary.json", summary)
    lines = ["# Codex GPT-5.5 comparison", "", "4 cases x 4 memory conditions.",
             "Medical prompts and application tools are preserved. Uses ChatGPT quota through Codex.",
             "Frozen Mem0 recall; no write-back. A new provider experiment, not a continuation of Gemini scores.", ""]
    for row in rows:
        lines += [f"## {row['case_id']} · {row['arm']}", "", row["question"], "",
                  "模型实际收到的上下文：", "", "```json",
                  json.dumps(row.get("context_actually_supplied"), ensure_ascii=False, indent=2), "```", "",
                  "实际回答：", "", row.get("answer", row.get("error", "未完成")), ""]
    (ROOT / "Codex原始对照记录.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


async def main(args):
    logger.remove()
    logger.add(sys.stderr, level="ERROR", diagnose=False, backtrace=False)
    if args.report_only:
        report()
        return
    items = ablation.prepare_inputs()
    protocol = {"model": MODEL, "provider": "Codex CLI; ChatGPT account",
                "medical_prompts_unchanged": True, "whatsapp_service_modified_or_started": False,
                "arms": ablation.ARMS, "selected_sources": [{"case_id": i["case_id"], "sha256": i["source_sha256"]} for i in items],
                "reasoning_effort": "low", "max_requests": 50,
                "cost_status": "ChatGPT quota; no Gemini or OpenAI API key",
                "tool_transport": "Structured next assistant turn; application executes requested tools",
                "source_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in
                                  [PROJECT / "evals/campaign_run.py", PROJECT / "evals/campaign_gateway.py",
                                   PROJECT / "evals/campaign_memory_ablation.py", PROJECT / "evals/campaign_gemini_memory.py"]}}
    path = ROOT / "protocol.json"
    if path.exists() and ablation.read(path) != protocol:
        raise ValueError("Frozen Codex protocol changed; create a new experiment")
    dump(path, protocol)
    gateway = CodexGateway(embedding_gateway=Gateway(OUTPUT, limit=4))
    ablation.ROOT = ROOT
    try:
        await tool_probe(gateway)
        if not args.probe_only:
            for index, item in enumerate(items):
                offset = (index + 2) % len(ablation.ARMS)
                for arm in ablation.ARMS[offset:] + ablation.ARMS[:offset]:
                    await ablation.execute(gateway, item, MODEL, arm)
    finally:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    asyncio.run(main(parser.parse_args()))
