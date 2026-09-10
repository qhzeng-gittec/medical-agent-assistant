"""Record deterministic product contracts and explicit defect probes without paid calls."""

import asyncio
import json
import subprocess
import sys

from campaign_gateway import dump
from campaign_run import OUTPUT, PROJECT
from core.skill_loader import load_skill_function
from run_components import run
from dataset_tools import DEFAULT_DATASET


class EmptyKnowledgeBase:
    def search(self, **kwargs):
        return []


class FailedKnowledgeBase:
    def search(self, **kwargs):
        raise TimeoutError("Controlled probe: retrieval unavailable")


async def main():
    contracts = await run(DEFAULT_DATASET)
    dump(OUTPUT / "component_contracts.json", contracts)
    risk = load_skill_function("analyze-symptoms", "symptoms", "analyze_symptoms", PROJECT)
    probes = []
    for question, kb, desired in [
        ("没有胸痛，没有呼吸困难，只有咳嗽", EmptyKnowledgeBase(), "不能把否认的胸痛/呼吸困难识别为已出现的高危症状"),
        ("最近不舒服", EmptyKnowledgeBase(), "未知不是低危：风险资料不足应显式表达不确定"),
        ("咳嗽", FailedKnowledgeBase(), "检索异常应可观察，不能返回宣称已使用知识库的成功型结果"),
    ]:
        risk.__globals__["_kb_instance"] = kb
        observed = await risk(question)
        probes.append({"input": question, "desired_behavior": desired, "observed": observed,
                       "assessment": "known_rule_or_error_observability_gap", "uses_llm": False})
    dump(OUTPUT / "risk_rule_probes.json", probes)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "evals/test_dataset_tools.py",
                             "evals/test_campaign.py", "-q"], cwd=PROJECT, capture_output=True, text=True)
    dump(OUTPUT / "automated_test_result.json", {"command": result.args, "returncode": result.returncode,
                                               "stdout": result.stdout, "stderr": result.stderr})
    print(json.dumps({"component_contracts": f"{contracts['passed']}/{contracts['total']}",
                      "automated_tests_exit": result.returncode, "risk_probes": len(probes)}, ensure_ascii=False))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    asyncio.run(main())
