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
    "api_key": os.getenv("MEM0_API_KEY", ""),
    "app_id": "medix-agent-swarm",
    "threshold": 0.3,
}
