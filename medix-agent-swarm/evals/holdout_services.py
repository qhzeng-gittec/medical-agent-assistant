"""Measured provider and memory adapters for the frozen holdout, not product fixes."""

import asyncio
import copy
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

from campaign_gateway import EMBED_MODEL, dump, sha
from campaign_rag import LocalRAG


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "evals/campaign_v1"
MODELS = ["minimax/minimax-m2.5", "qwen/qwen3.5-27b", "gemini-3.5-flash"]
GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
TARGET_ROLES = {"supervisor", "diagnostic_agent", "consultation_agent", "research_agent"}
SERVICE_CONFIG = {
    "models": MODELS, "embedding_model": EMBED_MODEL, "temperature": 0.2,
    "target_max_output_tokens": 8192, "request_timeout_seconds": 180,
    "transient_http_max_attempts": 3, "budget_cap_usd": None,
    "budget_authorization": "User explicitly removed previous cap on 2026-09-08",
    "mem0_threshold": 0.3, "mem0_observation_timeout_seconds": 120,
    "mem0_observation": "Settle accepted writes and list actual records between user turns",
    "seed_policy": "Actual product user replay for profile; original authored dialogue via real Mem0 add for events",
    "controlled_retrieval": "Authored candidate pool, no semantic ranking; not a real-retrieval score",
    "live_rag_corpus": "Existing frozen 15-chunk corpus; no holdout-driven corpus expansion",
    "external_deep_research": "Explicit not-configured error; not fake web search",
    "fine_tuned_medical_model": "Not served; no fine-tuned endpoint configured",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def runtime_config(output=None):
    config = copy.deepcopy(SERVICE_CONFIG)
    config["corpus_sha256"] = sha(DATA / "corpus.jsonl")
    if output is not None:
        config["artifacts"] = {name: sha(Path(output) / name)
                               for name in ("corpus_vectors.json", "provider_catalog.json")}
    return config


class HoldoutGateway:
    """One gateway per process, separate ledgers, bounded retries, no former spend cap."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.google_key = os.environ["MEDIX_GEMINI_API_KEY"]
        if os.environ.get("MEDIX_GEMINI_BASE_URL", "").rstrip("/") != GOOGLE_BASE:
            raise ValueError("Unrecognized Google endpoint; no credential sent")
        if os.environ.get("MEDIX_GEMINI_MODEL") != MODELS[2]:
            raise ValueError("Configured Google model differs from frozen model")
        self.lock = threading.RLock()
        self.spent = 0.0
        self.halted = {}
        self.instance = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.ledger = self.root / "ledgers" / f"api-{self.instance}.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        catalog_path = self.root / "provider_catalog.json"
        if catalog_path.exists():
            self.catalog = read(catalog_path)
        else:
            self.catalog = {}
            for endpoint in ("models", "embeddings/models"):
                response = httpx.get(f"https://openrouter.ai/api/v1/{endpoint}", timeout=45)
                response.raise_for_status()
                self.catalog.update({m["id"]: m for m in response.json()["data"]
                                     if m["id"] in MODELS[:2] + [EMBED_MODEL]})
            if set(MODELS[:2] + [EMBED_MODEL]) - self.catalog.keys():
                raise ValueError("An evaluation model is absent from the current official catalog")
            self.catalog[MODELS[2]] = {"id": MODELS[2], "provider": "google_direct",
                "pricing": {"prompt": "0.0000015", "completion": "0.000009"},
                "cost_status": "estimated_standard_list_price_not_invoice",
                "price_source": "https://ai.google.dev/gemini-api/docs/pricing"}
            dump(catalog_path, self.catalog)

    def redact(self, text):
        for key in (self.key, self.google_key, os.getenv("MEM0_API_KEY")):
            if key:
                text = str(text).replace(key, "[REDACTED]")
        return str(text)

    def append(self, entry):
        with self.lock, self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def request(self, endpoint, payload, trace, role):
        model = payload["model"]
        google = model == MODELS[2]
        provider = "google_direct" if google else "openrouter"
        if model not in self.catalog or endpoint not in {"embeddings", "chat/completions"}:
            raise ValueError("Request outside the frozen provider configuration")
        if google and endpoint != "chat/completions":
            raise ValueError("Google is not used for embeddings")
        halt_path = self.root / f"provider_halt_{provider}.json"
        if halt_path.exists():
            raise RuntimeError(f"{provider} has an unresolved authentication/credit halt")
        url = (GOOGLE_BASE if google else "https://openrouter.ai/api/v1") + "/" + endpoint
        key = self.google_key if google else self.key
        prices = self.catalog[model]["pricing"]
        for attempt in range(3):
            request_id = str(uuid.uuid4())
            event = {"event": "api_request", "request_id": request_id, "role": role,
                     "provider": provider, "endpoint": endpoint, "attempt": attempt + 1,
                     "payload": copy.deepcopy(payload), "cost_usd": 0.0}
            trace.append(event)
            started = time.monotonic()
            self.append({"request_id": request_id, "model": model, "provider": provider,
                         "role": role, "status": "started", "cost_usd": None})
            retry = False
            try:
                response = httpx.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=180)
                if response.is_error:
                    event["provider_error_body"] = self.redact(response.text[:2000])
                    event["http_status"] = response.status_code
                    event["cost_status"] = "http_error_cost_not_reported"
                    if response.status_code in {401, 402}:
                        dump(halt_path, {"provider": provider, "http_status": response.status_code,
                                         "request_id": request_id, "reason": "Check credentials/credits before resuming"})
                    retry = response.status_code in {429, 500, 502, 503, 504} and attempt < 2
                response.raise_for_status()
                data = response.json()
                if data.get("error"):
                    raise RuntimeError(self.redact(data["error"]))
                usage = data.get("usage", {})
                if usage.get("cost") is not None:
                    cost = float(usage["cost"])
                    cost_status = "provider_reported"
                elif usage.get("prompt_tokens") is not None:
                    completion = usage.get("completion_tokens", 0)
                    if google:
                        completion = max(completion, usage.get("total_tokens", 0) - usage["prompt_tokens"])
                    cost = usage["prompt_tokens"] * float(prices["prompt"]) + completion * float(prices.get("completion", 0))
                    cost_status = "estimated_standard_list_price_not_invoice" if google else "estimated_from_usage_and_catalog"
                else:
                    cost, cost_status = 0.0, "unknown_not_zero_cost"
                event.update(status="ok", cost_usd=cost, cost_status=cost_status,
                             response=data if endpoint != "embeddings" else {
                                 "model": data.get("model"), "usage": usage, "vector_count": len(data.get("data", []))})
                return data
            except Exception as error:
                event.update(status="error", error_type=type(error).__name__, error=self.redact(error))
                event.setdefault("cost_status", "unknown_not_zero_cost")
                if not retry:
                    raise
            finally:
                event["elapsed_seconds"] = round(time.monotonic() - started, 3)
                with self.lock:
                    self.spent += event["cost_usd"]
                    self.append({k: event[k] for k in ("request_id", "provider", "role", "status", "cost_usd", "cost_status", "elapsed_seconds")}
                                | {"model": model})
            # Only explicit transient HTTP responses are retried, never unknown write/timeout outcomes.
            time.sleep(2 ** (attempt + 1))

    async def chat(self, model, messages, trace, role, tools=None, tool_choice="auto", max_tokens=8192):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
        if model == MODELS[2]:
            payload["reasoning_effort"] = "low"
        else:
            payload.update(reasoning={"enabled": False} if role == "patient_selector" else {"effort": "low"},
                           provider={"require_parameters": True})
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        return await asyncio.to_thread(self.request, "chat/completions", payload, trace, role)


class FixtureKnowledge:
    def __init__(self, environment, trace):
        self.mode, self.docs, self.trace = environment["rag_mode"], environment["documents"], trace
        self.backend_label = self.mode

    def search(self, query, top_k=5, filter_type=None):
        if isinstance(top_k, bool) or int(top_k) != top_k or not 1 <= int(top_k) <= 10:
            raise ValueError("top_k must be an integer between 1 and 10")
        top_k = int(top_k)
        event = {"event": "retrieval", "query": query, "top_k": top_k,
                 "filter_type": filter_type, "status": self.mode, "documents": []}
        self.trace.append(event)
        if self.mode == "retrieval_timeout":
            raise TimeoutError("Controlled retrieval outage")
        if self.mode == "controlled":
            event["documents"] = [{**copy.deepcopy(doc), "score": 1.0} for doc in self.docs[:top_k]]
            event["ranking"] = "authored_order_not_similarity"
        return copy.deepcopy(event["documents"])


class FixtureMemory:
    """Explicit disabled mode or controlled read pool; never silently enables Mem0."""

    def __init__(self, environment, user_ids, trace):
        self.enabled = environment["memory_mode"] == "controlled"
        self.backend_label = "controlled_read_pool" if self.enabled else "disabled"
        self.records = copy.deepcopy(environment["memory_records"])
        self.user_ids, self.trace = user_ids, trace
        self.seed_replay = False

    def prepare_fixture(self, seed_history):
        self.trace.append({"event": "memory_fixture", "backend": self.backend_label,
                           "seed_history_count": len(seed_history), "record_count": len(self.records)})

    def search_similar_sessions(self, query, user_id, limit=3):
        if self.seed_replay or not self.enabled:
            return []
        items = [r for r in self.records if self.user_ids[r["user_key"]] == user_id][:limit]
        source_fields = {"user": "user_statement", "assistant": "assistant_response"}
        result = [{"memory_id": r["id"], "content": r["content"], "metadata":
                   {source_fields[r["source_role"]]: r["content"]} if r["source_role"] in source_fields else {}}
                  for r in items]
        self.trace.append({"event": "memory_search", "backend": self.backend_label, "query": query,
                           "user_id": user_id, "limit": limit, "memories": result,
                           "ranking": "authored_order_not_semantic_retrieval"})
        return copy.deepcopy(result)

    def add_session_summary(self, **kwargs):
        self.trace.append({"event": "memory_write_suppressed", "backend": self.backend_label,
                           "seed_replay": self.seed_replay, "reason": "Read-side fixture or disabled-memory condition"})
        return None

    def settle(self):
        return {"status": "not_applicable", "backend": self.backend_label}

    def snapshot(self, user_id):
        return {"backend": self.backend_label, "write_observation_available": False,
                "memories": [copy.deepcopy(r) for r in self.records if self.user_ids[r["user_key"]] == user_id]}


class RecordingMemoryClient:
    def __init__(self, owner):
        self.owner = owner

    def add(self, **kwargs):
        result = self.owner.call("add", kwargs, lambda: self.owner.sdk.add(**kwargs))
        self.owner.pending.append(copy.deepcopy(result))
        return result

    def search(self, **kwargs):
        return self.owner.call("search", kwargs, lambda: self.owner.sdk.search(**kwargs))


class LiveMemory:
    enabled = True
    backend_label = "live_mem0_platform"

    def __init__(self, environment, user_ids, trace, run_dir):
        os.environ["MEM0_TELEMETRY"] = "false"
        from mem0 import MemoryClient
        from memory.long_term import LongTermMemory
        self.environment, self.user_ids, self.trace = environment, user_ids, trace
        self.run_dir = Path(run_dir)
        self.key = os.environ["MEM0_API_KEY"]
        self.app_id = "medix-holdout-" + hashlib.sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:20]
        self.sdk = MemoryClient(api_key=self.key, client=httpx.Client(timeout=45))
        self.pending, self.sequence = [], 0
        self.seed_replay = False
        self.lock = threading.RLock()
        self.product = LongTermMemory(config={"app_id": self.app_id, "threshold": .3}, client=RecordingMemoryClient(self))
        dump(self.run_dir / "mem0_identity.json", {"app_id": self.app_id, "user_ids": user_ids,
             "synthetic": True, "delete_operations": False, "mem0_monetary_cost": "not_exposed"})

    def call(self, operation, params, fn):
        with self.lock:
            self.sequence += 1
            path = self.run_dir / "mem0_operations" / f"{self.sequence:04d}_{operation}.json"
        event = {"event": "mem0_operation", "operation": operation, "parameters": copy.deepcopy(params),
                 "status": "started", "backend": self.backend_label}
        self.trace.append(event)
        dump(path, event)
        started = time.monotonic()
        try:
            result = fn()
            event.update(status="completed", result=result)
            return result
        except Exception as error:
            event.update(status="error", error_type=type(error).__name__, error=str(error).replace(self.key, "[REDACTED]"))
            raise
        finally:
            event["seconds"] = time.monotonic() - started
            dump(path, event)

    def prepare_fixture(self, seed_history):
        # Use product's own current extraction instruction, captured without any network write.
        from memory.long_term import LongTermMemory
        captured = {}

        class Capture:
            def add(self, **kwargs):
                captured.update(kwargs)
                return {"results": []}

        template = LongTermMemory(config={"app_id": self.app_id}, client=Capture())
        for index, record in enumerate(seed_history):
            user = self.user_ids[record["user_key"]]
            session = user + "-authored-seed-" + str(index)
            template.add_session_summary(user, session, "", "")
            params = copy.deepcopy(captured)
            params["messages"] = copy.deepcopy(record["messages"])
            params["metadata"]["user_statement"] = "\n".join(m["content"] for m in record["messages"] if m["role"] == "user")[:1000]
            params["metadata"]["assistant_response"] = "\n".join(m["content"] for m in record["messages"] if m["role"] == "assistant")[:1000]
            self.trace.append({"event": "seed_source_provenance", "seed_session": record["session"],
                               "source": "authored_historical_dialogue_not_generated_bootstrap"})
            receipt = self.call("seed_add", params, lambda p=params: self.sdk.add(**p))
            self.pending.append(receipt)
        # Authored memory records are observations for controlled conditions, never hand-inserted ideal live memories.
        if self.environment["memory_records"]:
            self.trace.append({"event": "live_fixture_records_not_injected", "count": len(self.environment["memory_records"]),
                               "reason": "Live Mem0 must extract from seed dialogue, not receive ideal memories"})

    def search_similar_sessions(self, query, user_id, limit=3):
        if self.seed_replay:
            return []
        return self.product.search_similar_sessions(query, user_id, limit)

    def add_session_summary(self, *args, **kwargs):
        if self.seed_replay:
            self.trace.append({"event": "memory_write_suppressed", "backend": self.backend_label,
                               "reason": "Profile bootstrap; authored event history is seeded separately"})
            return None
        return self.product.add_session_summary(*args, **kwargs)

    def settle(self):
        started = time.monotonic()
        completed = []
        for receipt in self.pending:
            if receipt.get("status") == "SUCCEEDED" or ("results" in receipt and not receipt.get("event_id")):
                completed.append(receipt)
                continue
            event_id = receipt.get("event_id")
            if not event_id:
                raise ValueError("Unknown Mem0 write receipt; completion cannot be assumed")
            deadline = time.monotonic() + 120
            delay = 0
            while True:
                if delay:
                    time.sleep(delay)

                def fetch():
                    response = self.sdk.client.get(f"/v1/event/{quote(str(event_id), safe='')}/")
                    response.raise_for_status()
                    return response.json()

                event = self.call("event", {"event_id": event_id}, fetch)
                if event["status"] == "SUCCEEDED":
                    completed.append(event)
                    break
                if event["status"] == "FAILED":
                    raise RuntimeError(f"Mem0 extraction failed: {event.get('error')}")
                if event["status"] not in {"PENDING", "RUNNING"}:
                    raise ValueError("Unrecognized Mem0 event status")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Mem0 write not settled within observation window")
                delay = min(10, delay + 2)
        self.pending.clear()
        result = {"status": "settled", "events": len(completed), "seconds": time.monotonic() - started}
        self.trace.append({"event": "memory_settle", **result})
        return result

    def snapshot(self, user_id):
        params = {"filters": {"AND": [{"user_id": user_id}, {"app_id": self.app_id}]}, "page_size": 100}
        data = self.call("list", params, lambda: self.sdk.get_all(**params))
        if isinstance(data, dict) and data.get("next"):
            raise RuntimeError("Mem0 snapshot is paginated; refusing to label a partial snapshot complete")
        items = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("Unexpected Mem0 snapshot envelope")
        return {"backend": self.backend_label, "write_observation_available": True, "memories": items}


def build_environment(gateway, environment, user_ids, trace, run_dir):
    if environment["rag_mode"] == "live_rag":
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
        kb.backend_label = "live_qwen_api_embeddings_local_cosine"
    else:
        kb = FixtureKnowledge(environment, trace)
    memory = (LiveMemory(environment, user_ids, trace, run_dir) if environment["memory_mode"] == "live_mem0"
              else FixtureMemory(environment, user_ids, trace))
    return kb, memory


async def preflight(root):
    """Public calibration only; no holdout questions are read."""
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(1, str(PROJECT.parent))
    from campaign_gateway import ModelAdapter
    gateway = HoldoutGateway(root)
    rows = []
    for model in MODELS:
        trace = []
        adapter = ModelAdapter(gateway, model, trace, "probe")
        tools = [{"type": "function", "function": {"name": "echo", "description": "Return a provided value",
                 "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                "required": ["value"], "additionalProperties": False}}}]
        messages = [{"role": "user", "content": "Call echo with value OK and then reply with its returned value."}]
        try:
            first = await adapter.chat_with_tools(messages, tools, "required")
            if not first.tool_calls:
                raise ValueError("Calibration did not produce a tool call")
            messages.append(first.wire_message)
            for call in first.tool_calls:
                if call.name != "echo" or call.arguments != {"value": "OK"}:
                    raise ValueError("Unexpected calibration tool arguments")
                messages.append(adapter.create_tool_message(call.id, call.name, {"value": "OK"}))
            final = await adapter.chat_with_tools(messages)
            if final.tool_calls or not final.content or "OK" not in final.content:
                raise ValueError("Calibration did not finish")
            rows.append({"model": model, "status": "passed"})
        except Exception as error:
            rows.append({"model": model, "status": "failed", "error": gateway.redact(error)})
        dump(Path(root) / "calibration" / f"{model.replace('/', '_')}.json", trace)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    # New cache is generated by the remote API; documents and queries never use a local embedding model.
    trace = []
    kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
    docs = kb.search("一般健康咨询的证据来源", top_k=2)
    dump(Path(root) / "calibration/embedding.json", {"document_count": len(kb.docs), "returned": len(docs), "trace": trace})
    os.environ["MEM0_TELEMETRY"] = "false"
    from mem0 import MemoryClient
    client = MemoryClient(api_key=os.environ["MEM0_API_KEY"], client=httpx.Client(timeout=45))
    dump(Path(root) / "calibration/mem0_connection.json", {"authenticated": True, "backend": "Mem0 Platform"})
    client.client.close()
    dump(Path(root) / "preflight.json", {"models": rows, "embedding_passed": True, "mem0_authenticated": True,
                                        "service_config": SERVICE_CONFIG})
    if any(row["status"] != "passed" for row in rows):
        raise RuntimeError("Preflight contains a provider failure; inspect before launching target runs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(preflight(args.output))
