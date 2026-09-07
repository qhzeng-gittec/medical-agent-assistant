"""Export a readable, actual two-turn trace, without inventing an execution example."""

import copy
import json

from campaign_gateway import dump
from campaign_grade import TARGET_ROLES
from campaign_run import OUTPUT


def main():
    run_id = "CONSULT-12__minimax_minimax-m2.5__r3"
    trace = json.loads((OUTPUT / "traces" / f"{run_id}.json").read_text(encoding="utf-8"))
    run = json.loads((OUTPUT / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
    steps, observations, turn = [], [], 0
    for event in trace:
        if event["event"] == "turn_start":
            turn = event["turn"]
        if event["event"] == "tool_observation":
            observations.append({"turn": turn, "agent": event["role"], "tool": event["tool_name"],
                                 "tool_call_id": event["tool_call_id"], "result": event["result"]})
        if event["event"] != "api_request" or event["role"] not in TARGET_ROLES:
            continue
        messages = copy.deepcopy(event["payload"]["messages"])
        for message in messages:
            message.pop("reasoning", None)
            message.pop("reasoning_details", None)
        response = event.get("response", {})
        returned_calls = [c["function"]["name"] for choice in response.get("choices", [])
                          for c in choice["message"].get("tool_calls", [])]
        steps.append({"step": len(steps) + 1, "user_turn": turn, "receiver": event["role"],
                      "message_count": len(messages), "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                      "content_characters": sum(len(str(m.get("content") or "")) for m in messages),
                      "requested_tools": returned_calls, "messages": messages})
    dump(OUTPUT / "实际上下文增长示例.json", {"source_run": run_id,
        "note": "All messages copied from actual API payloads. Only provider reasoning fields omitted for readability; original trace retains them.",
        "dialogue": [{"user": t["user"], "answer": t["answer"]} for t in run["turns"]],
        "model_requests": steps, "tool_observations": observations})
    paragraphs = ["# 一条真实问诊的上下文怎样增长", "",
        f"来自实际记录 `{run_id}`，不是手工编出来的流程。", "",
        f"第一轮：{run['turns'][0]['user']}", "",
        f"第二轮：{run['turns'][1]['user']}", "",
        "下表的消息数指发送给模型的 messages 数组长度。JSON 字符串内还可能嵌着病例事实和 recent_history，"
        "所以 2 条消息不代表只知道两句话。token 是 API 回报的该次输入 token，不能与字符数混用。", "",
        "| 步骤 | 用户轮次 | 接收方 | messages 数 | 输入 token | 本次请求工具 |", "|---|---|---|---:|---:|---|"]
    for step in steps:
        paragraphs.append(f"| {step['step']} | {step['user_turn']} | {step['receiver']} | {step['message_count']} | "
                          f"{step['prompt_tokens']} | {', '.join(step['requested_tools']) or '无，输出文本'} |")
    paragraphs.extend(["", "## 怎么读对应 JSON", "",
        "1. 找到 model_requests：每一项就是一次真实模型调用的输入。该子 Agent 首次调用通常只有 system 和 user；"
        "user.content 里的 context 才是总控分给它的事实和之前 Agent 的最终发现。",
        "2. 工具调用后，看同一个子 Agent 下一次模型请求：会追加 assistant.tool_calls 和对应的 tool 结果，"
        "工具结果里的 documents 首次出现带 content，重复块可能只带 reference.tool_call_id。",
        "3. 返回总控后，总控的 tool 结果只带子 Agent 最终 answer，不带那个子 Agent 的完整检索对话。",
        "4. 第二个用户轮次，总控又从 system + user 开始构造 messages。user.content.context.recent_history "
        "包含上一轮用户问题和最后答复，不是直接把上一轮总控/子 Agent 的完整工具链续进去。",
        "5. 因此，本例第二轮能复述的依据必须存在于上一轮最终答复或其他真正传入的记忆中；"
        "子 Agent 曾经看到但没有进入最终答复的信息，不会自动神奇地出现。", "",
        "完整可读输入见同目录 `实际上下文增长示例.json`。为了可读性仅省略 provider reasoning 字段；"
        "未经省略的原始 API 请求保存在 traces 目录。本示例没有修改系统的记忆实现。"])
    (OUTPUT / "实际上下文增长示例.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    print({"example": run_id, "actual_model_requests": len(steps), "tool_observations": len(observations)})


if __name__ == "__main__":
    main()
