"""Copy to config.py at the repository root and set environment variables."""

import os

LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model_name": os.getenv("LLM_MODEL", ""),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    "temperature": 0.7,
    "max_tokens": 8192,
}

MEM0_CONFIG = {
    "app_id": "medix-agent-swarm",
    "storage_path": ".mem0",
    "llm_model": "qwen/qwen3.5-27b",
    "embedding_model": "qwen/qwen3-embedding-8b",
    "embedding_dims": 4096,
    "threshold": 0.3,
}
# Optional local memory reads OPENROUTER_API_KEY; MEM0_LOCAL_PATH overrides storage_path.
