"""Derive version-separated report metrics from existing records; no API calls."""

import hashlib
import json
from pathlib import Path


EVALS = Path(__file__).resolve().parent
OLD = EVALS / "results/full_system_v1"
NEW = EVALS / "results/improvement_v2"
OUT = EVALS / "results/metric_snapshot_20260908"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    source_paths = [OLD / "campaign_summary.json", OLD / "rag_boundary_v1/summary.json",
                    OLD / "rag_near_miss_v1/summary.json", OLD / "mem0_repeat_summary/summary.json",
                    NEW / "summary.json", NEW / "verification.json"]
    primary, rag, near, mem0, paired, verification = [read(p) for p in source_paths]
    old_models = []
    for row in primary["model_stats"]:
        n = row["tasks"]
        old_models.append({"model": row["model"], "cases": n,
                           "both_judges_all_pass": row["both_judges_all_pass_cases"],
                           "both_judges_all_pass_pct": 100 * row["both_judges_all_pass_cases"] / n,
                           "any_judge_critical_failure": row["any_judge_critical_failure_cases"],
                           "any_judge_critical_failure_pct": 100 * row["any_judge_critical_failure_cases"] / n,
                           "turns": row["completed_turns"], "mean_turn_seconds": row["mean_turn_seconds"],
                           "p95_turn_seconds": row["p95_turn_seconds"]})
    write_checks = []
    for model in sorted({g["model"] for g in paired["groups"]}):
        for variant in ("baseline", "candidate"):
            errors = 0
            for cid in ("N01", "N02", "N06"):
                path = NEW / "runs" / f"{cid}__{model.replace('/', '_')}__{variant}.json"
                source_paths.append(path)
                row = read(path)
                assert row["status"] == "completed"
                profile = row["turns"][-1]["profile_after"]
                categories = ("conditions", "medications") if cid == "N02" else ("allergies",)
                errors += any(f["status"] == "active" for category in categories for f in profile.get(category, []))
            write_checks.append({"model": model, "variant": variant, "cases": 3, "erroneous_active_writes": errors})
    base_rag = next(r for r in rag["threshold_sweep"] if r["k"] == 3 and r["threshold"] == 0)
    near_rag = next(r for r in near["threshold_sweep"] if r["k"] == 3 and r["threshold"] == .4)
    result = {"new_paid_requests": 0, "all_results_are_development_or_regression": True,
              "clinical_accuracy_available": False, "independent_holdout_score_available": False,
              "old_primary": old_models,
              "judge_disagreement": {"disagreements": primary["judge_disagreement_criteria"],
                                     "compared_criteria": primary["judge_comparable_criteria"],
                                     "percent": 100 * primary["judge_disagreement_criteria"] / primary["judge_comparable_criteria"]},
              "old_rag": {"base_probes": rag["completed"], "near_miss_probes": near["completed"],
                          "top3_unthresholded": base_rag, "near_miss_top3_threshold04": near_rag},
              "old_mem0_repetitions": mem0["repetitions"], "paired_execution": paired,
              "paired_erroneous_active_write_checks": write_checks,
              "cumulative_accounted_usd": verification["combined_accounted_usd"],
              "source_hashes": {str(p.relative_to(EVALS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 可以写进报告的数字，以及不能宣称的结论", "",
             "2026-09-08 从已有原始记录重新统计；没有新调用付费 API。旧版问诊主测与新版专项对照分开列出，不能混成当前版本总分。", "",
             "## 1. 旧版完整问诊开发主测", "",
             "24 个原始场景，每模型一次。两位自动评审的结论未经独立临床专家确认，以下不是医疗准确率。", "",
             "| 模型 | 双评审全部通过 | 至少一位评审判关键失败 | 用户轮次 | 平均最终回复秒数 | P95 最终回复秒数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in old_models:
        lines.append(f"| {row['model']} | {row['both_judges_all_pass']}/{row['cases']}（{row['both_judges_all_pass_pct']:.1f}%） | "
                     f"{row['any_judge_critical_failure']}/{row['cases']}（{row['any_judge_critical_failure_pct']:.1f}%） | "
                     f"{row['turns']} | {row['mean_turn_seconds']:.2f} | {row['p95_turn_seconds']:.2f} |")
    lines += ["", f"自动评审分歧：{primary['judge_disagreement_criteria']}/{primary['judge_comparable_criteria']} 个可比条目"
              f"（{result['judge_disagreement']['percent']:.1f}%）。同意也不等于正确。P95 来自很小且同一病例内相关的轮次样本，只用于描述本批观测。",
              "", "## 2. 最近八题改前/改后专项", "",
              "8 个场景 × 3 模型 × 2 版本 = 48 组执行，66 轮用户对话；不是 48 个独立病例。", "",
              "最稳妥的共同指标是三类不应产生本人阳性事实的任务：询问、他人信息、引用翻译。每模型同样三题，旧版均为 3/3 次错误阳性写入，新版均为 0/3。合计 9/9 → 0/9 是重复执行口径，不是 9 个独立病例或总体零错误率。", "",
              "| 模型 | 版本 | 模型请求 | 输入 Token | 每案例平均秒数 |",
              "|---|---|---:|---:|---:|"]
    for row in paired["groups"]:
        lines.append(f"| {row['model']} | {row['variant']} | {row['model_requests']} | {row['prompt_tokens']} | {row['case_mean_seconds']:.2f} |")
    lines += ["", "这里的案例含一至三轮，不能与上一节单轮耗时直接比较。档案原始检查有药名子串误判、强制保存否认的策略偏向和只检查最终状态的问题，因此不把 7/8、8/8 包装成新版准确率。", "",
              "## 3. RAG：召回与答案依据必须分开", "",
              f"原基础集 {rag['completed']} 个探针，其中 40 个正常可回答查询；15 个文档块，全部向量由 OpenRouter Qwen API 生成。无阈值 Top 3 的至少一组证据命中 {base_rag['any_hit_n']}/40，全部预设证据组命中 {base_rag['all_hit_n']}/40（97.5%）。这是原标签分数，其中一题存在等价证据标注争议，不能反推临床回答准确率。", "",
              f"更值得报告的失败是：追加的 12 个近义但无充分答案的问题，在阈值 0.4 下仍有 {near_rag['unanswerable_returned_n']}/12（91.7%）返回候选。这不表示召回器有 91.7% 医学错误，而是说明相似候选不等于回答所需证据。追加集看过结果后构造，只能称开发反例。", "",
              "## 4. Mem0：真实云服务组件，不冒充整个问诊链路", "",
              "12 类场景在三个独立用户/应用空间重复，36 组执行；57 次写入、255 次检索。每次 17 个有预设词组的查询，Top 3 原始词组覆盖为 16/17、15/17、15/17；后两次各有一处英文否定表达导致评分漏识别。它们不是经过语义复核的召回率，不能直接相加当 51 个独立案例。", "",
              "另有最近 3 组云端探针和来源元数据检查。主动历史补查八题对照使用受控记录池，不测云端语义检索质量。", "",
              "## 5. 为什么当前还不够完善", "",
              "- 场景已用于调试，缺独立未暴露测试；同一主题多次执行不是新增病例。",
              "- 问诊主测、RAG、Mem0 没有在同一冻结新版上完成完整在线端到端矩阵。",
              "- 缺少足够普通成功场景、多个医学主题、长历史及反复更正、真实并发、模型重复采样。",
              "- 缺单 Agent 对照、微调医疗模型服务接入、外部检索与来源忠实性的完整测评。",
              "- 语义评分器也会错，需要双盲评审、分歧复核和独立临床审核；字符串出现不代表含义正确。", "",
              "## 6. 新测试集的预注册指标", "",
              "后续新集按病例家族统计，不把重复采样视为独立样本。各指标都保留分子、分母、适用条件、不确定和未执行数量，不用一个加权总分掩盖关键失败。", "",
              "| 维度 | 应报告的数字 | 判定依据 |",
              "|---|---|---|",
              "| 问诊任务 | 场景完成率、关键失败率、必要追问覆盖率 | 逐题冻结语义检查；不强制完整问诊模板 |",
              "| 档案 | 错误本人事实写入率、应保存事实遗漏率、逐轮更正成功率 | 每轮真实持久状态；不要求每个否认都新增 |",
              "| 事件记忆 | 写入保真、来源/时间/主体保真、召回覆盖、最终使用正确率 | 五层分别核查；以实际存储内容为依据 |",
              "| 证据 | Recall@K、必要证据完整覆盖、引用支持率、无答案诚实率 | 召回候选与支持具体结论分开 |",
              "| 工具执行 | 虚假完成声明率、权限拒绝、工具配对违规、超限执行次数 | 回答与实际调用轨迹对照 |",
              "| 效率稳定性 | 每轮 Token/费用、P50/P95、首次可见安全提示时间、重复成功率 | 真实用户可见时刻；不以内部日志充当已告知 |", "",
              "本轮只整理现有指标、生成并审核新的隔离测试候选，不在这里宣称新集已经执行。新集与执行协议须冻结，评分先在独立校准集验证。", "",
              f"既有累计计费/保守预留约 {verification['combined_accounted_usd']:.2f} 美元，总上限仍为 5 美元；Google 按用量估价而非账单，Mem0 配额另计。本轮没有新增付费请求。"]
    (OUT / "报告指标与评测边界.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"old_primary_models": old_models, "write_checks": write_checks,
                      "output": str(OUT), "new_paid_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
