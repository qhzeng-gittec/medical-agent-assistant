"""Metered OpenRouter access for the evaluation campaign, not the product client."""

import asyncio
import copy
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import httpx
from codex_gateway import request as codex_request


MODELS = ["minimax/minimax-m2.5", "qwen/qwen3.5-27b", "gpt-5.5"]
EMBED_MODEL = "qwen/qwen3-embedding-8b"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BudgetExceeded(RuntimeError):
    pass


class ApiAccessBlocked(RuntimeError):
    pass


class Gateway:
    def __init__(self, root: Path, limit: float = 5.0):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.lock = threading.RLock()
        self.limit = limit
        self.ledger = root / "api_ledger.jsonl"
        self.spent = 0.00008073  # First connection probe, retained from the preceding turn.
        self.reserved = 0.0
        self.halt_reason = None
        if self.ledger.exists():
            latest = {}
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                latest[entry["request_id"]] = entry
            self.spent += sum(entry["charged_or_reserved_usd"] for entry in latest.values())
        self.catalog = {}
        for endpoint in ("models", "embeddings/models"):
            response = httpx.get(f"https://openrouter.ai/api/v1/{endpoint}", timeout=30)
            response.raise_for_status()
            for model in response.json()["data"]:
                if model["id"] in MODELS + [EMBED_MODEL]:
                    self.catalog[model["id"]] = model
        if set(MODELS[:2] + [EMBED_MODEL]) - self.catalog.keys():
            raise ValueError("Requested model missing from current catalog")
        self.catalog[MODELS[2]] = dict(id=MODELS[2],provider="codex_chatgpt",pricing={"prompt":"0","completion":"0"},cost_status="chatgpt_quota")
        dump(root / "models_snapshot.json", self.catalog)

    def _append(self, row: dict) -> None:
        with self.lock, self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def request(self, endpoint: str, payload: dict, trace: list, role: str) -> dict:
        if payload["model"] == MODELS[2]:
            if endpoint != "chat/completions":
                raise ValueError("Codex does not provide embeddings")
            return codex_request(self.root,payload,trace,role)
        prices = self.catalog[payload["model"]]["pricing"]
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        reserve = ((size * 2 + 2048) * float(prices["prompt"])
                   + payload.get("max_tokens", 0) * float(prices.get("completion", 0))) * 1.25
        with self.lock:
            if self.halt_reason:
                raise ApiAccessBlocked(self.halt_reason)
            if self.spent + self.reserved + reserve > self.limit:
                raise BudgetExceeded(f"Campaign spend plus reservation would exceed ${self.limit}")
            self.reserved += reserve
        request_id = str(uuid.uuid4())
        self._append({"request_id": request_id, "model": payload["model"], "role": role,
                      "charged_or_reserved_usd": reserve, "cost_status": "reserved_inflight",
                      "status": "started"})
        started = time.monotonic()
        event = {"event": "api_request", "request_id": request_id, "role": role,
                 "endpoint": endpoint, "payload": copy.deepcopy(payload)}
        trace.append(event)
        charged = reserve
        status = "unknown_cost_reserved"
        try:
            response = httpx.post(f"https://openrouter.ai/api/v1/{endpoint}",
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json=payload, timeout=75)
            if response.is_error:
                event["provider_error_body"] = response.text[:2000].replace(self.key, "[REDACTED]")
                if response.status_code in {401, 402}:
                    with self.lock:
                        self.halt_reason = f"OpenRouter HTTP {response.status_code}; queued paid requests paused. Check credentials/credits before resuming."
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Provider returned error: {data['error']}")
            usage = data.get("usage", {})
            actual = usage.get("cost")
            if actual is not None:
                charged, status = float(actual), "provider_reported"
            elif usage.get("prompt_tokens") is not None:
                charged = (usage["prompt_tokens"] * float(prices["prompt"])
                           + usage.get("completion_tokens", 0) * float(prices.get("completion", 0)))
                status = "estimated_from_usage_and_catalog"
            event.update({"response": data if endpoint != "embeddings" else {
                "model": data.get("model"), "usage": usage,
                "vector_count": len(data.get("data", []))}, "status": "ok"})
            return data
        except Exception as error:
            event.update({"status": "error", "error_type": type(error).__name__,
                          "error": str(error).replace(self.key, "[REDACTED]")})
            raise
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - started, 3)
            event["cost_usd"] = charged
            event["cost_status"] = status
            with self.lock:
                self.reserved -= reserve
                self.spent += charged
                self._append({"request_id": request_id, "model": payload["model"], "role": role,
                              "charged_or_reserved_usd": charged, "cost_status": status,
                              "elapsed_seconds": event["elapsed_seconds"], "status": event["status"]})

    async def chat(self, model: str, messages: list, trace: list, role: str,
                   tools=None, tool_choice="auto", max_tokens=2048) -> dict:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0.2, "reasoning": {"effort": "low"},
                   "provider": {"require_parameters": True}}
        if role == "patient_selector":
            payload["reasoning"] = {"enabled": False}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        return await asyncio.to_thread(self.request, "chat/completions", payload, trace, role)


class ModelAdapter:
    def __init__(self, gateway: Gateway, model: str, trace: list, role: str):
        self.gateway, self.model, self.trace, self.role = gateway, model, trace, role

    async def chat_with_tools(self, messages, tools=None, tool_choice="auto", **kwargs):
        from core.llm_client import LLMResponse, ToolCall
        data = await self.gateway.chat(self.model, messages, self.trace, self.role, tools, tool_choice)
        choice = data["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            raise RuntimeError(f"Incomplete model response: {choice['finish_reason']}")
        message = choice["message"]
        calls = [ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                 for c in message.get("tool_calls", [])]
        result = LLMResponse(message.get("content"), calls, choice["finish_reason"])
        result.wire_message = {k: copy.deepcopy(v) for k, v in message.items()
                               if k in {"role", "content", "tool_calls", "reasoning", "reasoning_details"}}
        return result

    def create_tool_message(self, tool_call_id, tool_name, result):
        message = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name,
                   "content": json.dumps(result, ensure_ascii=False)}
        self.trace.append({"event": "tool_observation", "role": self.role,
                           "tool_call_id": tool_call_id, "tool_name": tool_name,
                           "result": copy.deepcopy(result)})
        return message
