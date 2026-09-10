import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from agents import DiagnosticAgent, ResearchAgent
from core import LLMResponse
from core.skill_loader import load_skill_function
from research.evidence_synthesizer import EvidenceSynthesizer
from research import web_search


SYMPTOM_TOOLS = [("analyze-symptoms", "symptoms", "analyze_symptoms")]


@pytest.mark.parametrize("skill,script,name", SYMPTOM_TOOLS)
@pytest.mark.parametrize("description", [
    "没有胸痛，也没有呼吸困难；昨天是陪父亲就诊，请区分主体。",
    "两小时前开始像被重物压着，走几步就得停下喘口气，和以往不一样。",
    "题目假设患者高热不退，我自己没有这些症状，只需找相关资料。",
])
def test_symptom_tools_preserve_case_and_return_evidence_without_classification(monkeypatch, skill, script, name, description):
    class KB:
        def search(self, **kwargs):
            assert kwargs == {"query": description, "top_k": 3, "filter_type": None}
            return [{"id": "d1", "content": "候选原文，适用性待判断", "score": .1,
                     "metadata": {"source": "示例资料"}}]

    execute = load_skill_function(skill, script, name)
    monkeypatch.setitem(execute.__globals__, "_kb_instance", KB())
    result = asyncio.run(execute(symptoms=description))
    assert set(result) == {"status", "answer", "documents"}
    assert result["status"] == "candidates"
    assert result["documents"][0]["content"] == "候选原文，适用性待判断"


@pytest.mark.parametrize("skill,script,name", SYMPTOM_TOOLS)
def test_empty_retrieval_does_not_assign_low_risk_or_possible_diseases(monkeypatch, skill, script, name):
    class KB:
        def search(self, **kwargs):
            return []

    execute = load_skill_function(skill, script, name)
    monkeypatch.setitem(execute.__globals__, "_kb_instance", KB())
    result = asyncio.run(execute(symptoms="突然出现难以描述的不适"))
    assert result["status"] == "no_results"
    assert result["documents"] == []
    assert "risk_level" not in result and "possible_diseases" not in result


@pytest.mark.parametrize("skill,script,name", SYMPTOM_TOOLS)
def test_retrieval_failure_is_visible_to_the_agent(monkeypatch, skill, script, name):
    class KB:
        def search(self, **kwargs):
            raise RuntimeError("knowledge service unavailable")

    execute = load_skill_function(skill, script, name)
    monkeypatch.setitem(execute.__globals__, "_kb_instance", KB())
    agent = DiagnosticAgent(llm_client=AsyncMock())
    agent.skill_registry.skills[name]["function"] = execute
    result = asyncio.run(agent.execute_tool(name, {"symptoms": "胸痛"}))
    assert result["success"] is False
    assert "knowledge service unavailable" in result["error"]
    assert "risk_level" not in result


@pytest.mark.parametrize("agent_class,answer", [
    (DiagnosticAgent, "风险等级：暂不确定。否认胸痛，高血压为既往记录。"),
    (ResearchAgent, "资料没有提供A级证据，无法据此分配等级或确认冲突。"),
])
def test_worker_deliverables_are_not_classified_or_rewritten(agent_class, answer):
    llm = AsyncMock()
    llm.chat_with_tools.return_value = LLMResponse(answer, [], "stop")
    agent = agent_class(llm_client=llm)
    result = asyncio.run(agent.process({"question": "只整理已有信息"}))
    assert result == {"answer": answer, "iterations": 1, "agent_id": agent.agent_id, "evidence": []}


def test_web_search_uses_the_model_query_unchanged(monkeypatch):
    query = "Compare evidence in adults without prior symptoms"

    class DDGS:
        def text(self, actual, **kwargs):
            assert actual == query
            return [{"title": "source", "href": "https://example.com", "body": "excerpt"}]

    monkeypatch.setattr(web_search, "DDGS_AVAILABLE", True)
    monkeypatch.setattr(web_search, "DDGS", DDGS, raising=False)
    result = asyncio.run(web_search.WebSearchTool().search(query))
    assert len(result) == 1


@pytest.mark.parametrize("skill,script,name,parameter,kind", [
    ("disease-code", "code", "disease_code", "disease_name", "disease_classification"),
    ("clinical-guideline", "guideline", "clinical_guideline", "query", "clinical_guideline"),
    ("recommend-lifestyle", "lifestyle", "recommend_lifestyle", "diagnosis", "lifestyle"),
])
def test_specialist_retrieval_preserves_model_query_and_structured_filter(monkeypatch, skill, script, name, parameter, kind):
    query = "仅查询已排除其他原因后的相关资料，不能套用到另一人群"

    class KB:
        def search(self, **kwargs):
            assert kwargs["query"] == query
            assert kwargs["filter_type"] == kind
            return []

    execute = load_skill_function(skill, script, name)
    monkeypatch.setitem(execute.__globals__, "_kb_instance", KB())
    asyncio.run(execute(**{parameter: query}))


def synthesis_payload():
    return {
        "key_findings": ["未发现可以直接用于当前人群的结论"],
        "evidence_level": "unknown",
        "confidence": .4,
        "conflicts": ["没有一致结论：两项资料对适用人群的描述不同"],
        "summary": "不能因为文中提到A级就认定当前结论具有该等级。",
        "recommendations": [],
    }


def test_synthesis_preserves_model_fields_including_negated_conflicts():
    payload = synthesis_payload()
    llm = AsyncMock()
    llm.chat.return_value = json.dumps(payload, ensure_ascii=False)
    report = asyncio.run(EvidenceSynthesizer(llm).synthesize("比较两项资料"))
    assert report.evidence_level == "unknown"
    assert report.conflicts == payload["conflicts"]
    assert report.summary == payload["summary"]
    assert report.sources == []


@pytest.mark.parametrize("change", [
    {"confidence": True}, {"confidence": 2}, {"conflicts": "没有"}, {"summary": ""},
])
def test_invalid_synthesis_schema_fails_explicitly(change):
    payload = {**synthesis_payload(), **change}
    llm = AsyncMock()
    llm.chat.return_value = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValueError):
        asyncio.run(EvidenceSynthesizer(llm).synthesize("测试"))


def test_malformed_synthesis_is_not_reported_as_success():
    llm = AsyncMock()
    llm.chat.return_value = "无法生成JSON"
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(EvidenceSynthesizer(llm).synthesize("测试"))
