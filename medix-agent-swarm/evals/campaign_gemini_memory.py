"""Direct Google comparison on the unchanged medical Agent and frozen recall cases."""

import argparse
import asyncio
import copy
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter

import httpx
from loguru import logger

import campaign_memory_ablation as ablation
from campaign_gateway import ApiAccessBlocked, BudgetExceeded, Gateway, ModelAdapter, dump, sha
from campaign_run import OUTPUT, PROJECT


ROOT = OUTPUT / "gemini_memory_ablation_v1"
MODEL = "gemini-3.5-flash"
INPUT_RATE = 1.50 / 1_000_000
OUTPUT_RATE = 9.00 / 1_000_000


class GoogleGateway:
    def __init__(self, root=ROOT, embedding_gateway=None, max_requests=50):
        self.root = OUTPUT  # Reuse the frozen, remote-Qwen-produced corpus vectors.
        self.key = os.environ["MEDIX_GEMINI_API_KEY"]
        self.base_url = os.environ["MEDIX_GEMINI_BASE_URL"].rstrip("/")
        if self.base_url != "https://generativelanguage.googleapis.com/v1beta/openai":
            raise ValueError("Unexpected Google endpoint; refusing to send the credential")
        if os.environ["MEDIX_GEMINI_MODEL"] != MODEL:
            raise ValueError("The selected project model changed; create a new experiment")
        self.embedding_gateway = embedding_gateway
        self.ledger = root / "google_api_ledger.jsonl"
        self.lock = threading.RLock()
        self.halt_reason = None
        self.reserved = 0.0
        self.limit = 1.0  # Other providers remain capped at $4, combined cap $5.
        self.spent = 0.025  # Unknown-cost reserve for the original-client connectivity probe.
        self.requests = 0
        self.max_requests = max_requests
        if self.ledger.exists():
            latest = {}
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                latest[row["request_id"]] = row
            self.spent += sum(r["charged_or_reserved_usd"] for r in latest.values())
            self.requests = len(latest)
            if any(r["status"] == "started" for r in latest.values()):
                raise RuntimeError("Unresolved Google request; inspect before resuming")

    def append(self, row):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def request(self, endpoint, payload, trace, role):
        if endpoint == "embeddings":
            if self.embedding_gateway is None:
                raise RuntimeError("Remote Qwen embedding gateway is not configured")
            return self.embedding_gateway.request(endpoint, payload, trace, role)
        if endpoint != "chat/completions" or payload["model"] != MODEL:
            raise ValueError("Request outside this comparison's scope")
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        reserve = ((size * 2 + 2048) * INPUT_RATE + payload["max_tokens"] * OUTPUT_RATE) * 1.25
        with self.lock:
            if self.halt_reason:
                raise ApiAccessBlocked(self.halt_reason)
            if self.spent + self.reserved + reserve > self.limit or self.requests >= self.max_requests:
                raise BudgetExceeded(f"Google comparison reached its ${self.limit} or {self.max_requests}-request cap")
            self.requests += 1
            self.reserved += reserve
        request_id = str(uuid.uuid4())
        entry = {"request_id": request_id, "model": MODEL, "provider": "google_direct", "role": role}
        self.append({**entry, "status": "started", "cost_status": "reserved_inflight", "charged_or_reserved_usd": reserve})
        event = {"event": "api_request", **entry, "endpoint": endpoint, "payload": copy.deepcopy(payload)}
        trace.append(event)
        started = time.monotonic()
        charged, cost_status = reserve, "unknown_cost_reserved"
        try:
            response = httpx.post(self.base_url + "/" + endpoint, json=payload,
                                  headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
            if response.is_error:
                event["provider_error_body"] = response.text[:2000].replace(self.key, "[REDACTED]")
                self.halt_reason = f"Google HTTP {response.status_code}; inspect before further requests"
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise RuntimeError(str(data["error"]).replace(self.key, "[REDACTED]"))
            usage = data.get("usage", {})
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                output_tokens = max(usage["completion_tokens"], usage.get("total_tokens", 0) - usage["prompt_tokens"])
                charged = usage["prompt_tokens"] * INPUT_RATE + output_tokens * OUTPUT_RATE
                cost_status = "estimated_standard_list_price_not_invoice"
            event.update(status="ok", response=data)
            return data
        except Exception as error:
            event.update(status="error", error_type=type(error).__name__, error=str(error).replace(self.key, "[REDACTED]"))
            raise
        finally:
            event.update(elapsed_seconds=round(time.monotonic() - started, 3), cost_usd=charged, cost_status=cost_status)
            with self.lock:
                self.reserved -= reserve
                self.spent += charged
                self.append({**entry, "status": event["status"], "cost_status": cost_status,
                             "charged_or_reserved_usd": charged, "elapsed_seconds": event["elapsed_seconds"]})

    async def chat(self, model, messages, trace, role, tools=None, tool_choice="auto", max_tokens=2048):
        payload = {"model": model, "messages": messages, "temperature": 0.2,
                   "max_tokens": max_tokens, "reasoning_effort": "low"}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        return await asyncio.to_thread(self.request, "chat/completions", payload, trace, role)


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
        result["error"] = str(error).replace(gateway.key, "[REDACTED]")
        raise
    finally:
        dump(path, {**result, "trace": trace})
    print(json.dumps(result), flush=True)


def report():
    rows = [ablation.read(p) for p in sorted((ROOT / "runs").glob("*.json"))]
    summary = {"model": MODEL, "provider": "google_direct", "planned_runs": 16,
               "statuses": dict(Counter(r["status"] for r in rows)), "independent_source_cases": 4,
               "single_sample_per_arm": True, "cloud_recall": "frozen actual Mem0 r1 results",
               "cloud_write_back": False, "clinical_validation": False,
               "protocol_failure_count": sum(len(r["protocol"]["failures"]) for r in rows),
               "response_contract_failure_count": sum(len(r["protocol"]["response_contract_failures"]) for r in rows),
               "estimated_target_usd_not_invoice": sum(r["metrics"]["target_cost_usd"] for r in rows),
               "model_requests": sum(r["metrics"]["model_requests"] for r in rows),
               "retrieval_executions": sum(r["metrics"]["retrieval_backend_executions"] for r in rows)}
    dump(ROOT / "summary.json", summary)
    lines = ["# Gemini 原始对照记录", "", "4 个开发案例 × 4 种记忆条件，各执行一次。不是 16 个独立病例。",
             "使用 WhatsApp 项目的 Google 接口、模型和密钥，但使用医疗项目原提示词、工具及状态逻辑。"
             "不启用 WhatsApp 的 JSON 输出模式；请求 low 推理强度、温度 0.2、最多 2048 输出 token。"
             "不同提供商的 low 不代表相同推理计算量。", "",
             "Mem0 为已完成真实云检索的冻结快照，不在本实验重新检索或写回；历史档案仍由原规则从用户原话提取。"
             "短期会话彼此隔离。Google 费用按标准价估算，不代表实际账单或是否使用免费额度。", ""]
    for row in rows:
        lines += [f"## {row['case_id']} · {row['arm']}", "", row["question"], "",
                  "模型实际收到的上下文：", "", "```json",
                  json.dumps(row.get("context_actually_supplied"), ensure_ascii=False, indent=2), "```", "",
                  "实际回答：", "", row.get("answer", row.get("error", "未完成")), ""]
    (ROOT / "Gemini原始对照记录.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


async def main(args):
    logger.remove()
    logger.add(sys.stderr, level="ERROR", diagnose=False, backtrace=False)
    if args.report_only:
        report()
        return
    items = ablation.prepare_inputs()
    protocol = {"model": MODEL, "provider": "Google direct; WhatsApp project credential",
                "medical_prompts_unchanged": True, "whatsapp_service_modified_or_started": False,
                "arms": ablation.ARMS, "selected_sources": [{"case_id": i["case_id"], "sha256": i["source_sha256"]} for i in items],
                "temperature": 0.2, "max_tokens": 2048, "reasoning_effort": "low",
                "google_budget_usd": 1, "other_provider_budget_usd": 4, "combined_budget_usd": 5,
                "probe_unknown_cost_reserve_usd": 0.025,
                "pricing": {"input_per_million_usd": 1.50, "output_including_thinking_per_million_usd": 9.00,
                            "source": "https://ai.google.dev/gemini-api/docs/pricing?hl=en", "checked_date": "2026-09-07"},
                "source_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in
                                  [PROJECT / "evals/campaign_run.py", PROJECT / "evals/campaign_gateway.py",
                                   PROJECT / "evals/campaign_memory_ablation.py", PROJECT / "evals/campaign_gemini_memory.py"]}}
    path = ROOT / "protocol.json"
    if path.exists() and ablation.read(path) != protocol:
        raise ValueError("Frozen Gemini protocol changed; create a new experiment")
    dump(path, protocol)
    gateway = GoogleGateway(embedding_gateway=Gateway(OUTPUT, limit=4))
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
