"""Build reader-facing tables from completed, verified experiment artifacts."""
from collections import Counter
import json

from common import HERE, RESULTS, ROOT, V2, digest, lines, read, write


def fraction(n,d):
    return f'{n}/{d}（{n/d:.1%}）' if d else '—'


def main():
    retrieval = read(RESULTS/'retrieval/summary.json')
    answers = read(RESULTS/'answers/summary.json')
    memory = read(RESULTS/'memory_answers/summary.json')
    answer_protocol = read(RESULTS/'answers/protocol.json')
    paired = read(RESULTS/'answers/paired_comparison.json')
    audit = read(RESULTS/'judge_audit/summary.json')
    cost = read(RESULTS/'cost_summary.json')
    changed = read(RESULTS/'retrieval/changed_cases.json')
    hybrid = answer_protocol['selected_method']
    labels = {'dense':'纯向量','bm25_native':'BM25 原始查询','bm25_translated':'BM25 英文翻译查询',
              'hybrid_native':'向量 + BM25 / 原始查询','hybrid_translated':'向量 + BM25 / 英文翻译查询'}
    text = ['# 混合检索、证据充分性与本地 Mem0 回答测评','',
            '在同一份 **1,016 篇 MedlinePlus 健康主题摘要**上比较纯向量、BM25 与加权 RRF 混合检索，再测量模型根据召回材料生成的最终回答。'
            '保留原有 400 条查询，新增 100 个此前未作答案来源的主题、400 条查询，共 **800 条查询 / 320 个场景族**；开发集 160 条，测试集 640 条。','',
            '检索使用真实 Qwen3-Embedding-8B API 向量，回答模型为 Qwen3.5-27B，回答评审为 MiniMax M2.5。'
            '本地 Mem0 的 96 个测试用户另外生成 288 份回答，覆盖个人历史事实与未披露信息。','',
            '**实测结论：** 混合检索的 Top-3 完整来源覆盖从 408/432 增至 410/432，收益较小，配对区间包含零。'
            '本轮补齐了资料不足和最终回答检查，但自动评审存在误判；下面同时公开原始评分与独立复核，不能把通过率理解为临床准确率。','',
            '## 检索设计与结果','',
            'BM25 与向量使用相同文档正文。两路分别排序后以加权 Reciprocal Rank Fusion 融合，保留两路得分和来源。'
            '考虑英文语料与中文问题的语言差异，另测实际 API 查询翻译；翻译只接收问题，不接收原文、标准答案或配对英文题。'
            '该方法与 [Elastic 的混合检索说明](https://www.elastic.co/docs/solutions/search/hybrid-search)中的词法 / 向量排名融合思路一致。','',
            '| 方法 | 完整来源覆盖 @1 | @3 | @5 | @10 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for method,m in retrieval['splits']['test'].items():
        text.append('| '+labels[method]+' | '+' | '.join(fraction(m['complete'][str(k)],m['answerable_queries']) for k in [1,3,5,10])+' |')
    text += ['', '此表仅统计测试集 **432 条完全可回答查询**，要求每组目标来源至少命中一项；80 条部分可回答、112 条无答案、16 条缺历史问题分别评测。'
             'Top-K 预算相同且不施加相似度阈值，避免把 RRF 分数当作余弦相似度。目标来源覆盖与最终回答正确性分别计分。','',
             '| 测试集分层 | 纯向量 @3 | 原始查询混合 @3 | 翻译查询混合 @3 |',
             '| --- | ---: | ---: | ---: |']
    for stratum,label in [('suite/original','原有冻结测试集'),('suite/extension','新增主题测试集'),('language/en','英文查询'),('language/zh','中文查询')]:
        values = retrieval['strata']['test/'+stratum]
        text.append('| '+label+' | '+' | '.join(fraction(values[m]['complete']['3'],values[m]['answerable_queries']) for m in ['dense','hybrid_native','hybrid_translated'])+' |')
    text += ['', '新增完全可回答问题中，纯向量已达到 159/160，存在明显的上限效应。扩展增加了主题覆盖和证据不足探针，但单来源摘要问题仍较容易；样本数量增加不等于任务难度增加。','',
             '两种混合配置分别在开发集网格中选择：向量权重 0.25/0.5/0.75/0.9，RRF 常数 10/60，每路候选窗口 20/50。'
             '只优化开发集完整来源覆盖 @3；平局按预先规定的顺序选择，并在计算测试排名前保存参数。','']
    for mode,selection in retrieval['selected'].items():
        c = selection['selected']
        text.append(f'- {labels["hybrid_"+mode]}：向量权重 {c["dense_weight"]}，RRF 常数 {c["rank_constant"]}，每路窗口 {c["window"]}。')
    text += ['','相对纯向量的 Top-3 覆盖差值采用同场景族配对 bootstrap，4,000 次重采样；同源中英文问题不当作独立样本：','']
    for interval in retrieval['paired_top3']:
        lo,hi = interval['family_bootstrap_95']
        text.append(f'- {labels[interval["candidate"]]}：{interval["difference"]*100:+.2f} 个百分点，95% 区间 [{lo*100:+.2f}, {hi*100:+.2f}]。')
    text += ['','原有 400 条查询的额外消融保留在 [原始查询对照](baseline_retrieval/summary.json)和 [查询翻译对照](baseline_translated/summary.json)。'
             '等权融合、开发集选择与最终配置均有记录，不根据测试成绩替换配置。','',
             f'最终翻译混合配置在测试集 Top-3 中有 **{sum(c["hybrid_complete"] for c in changed)} 条改善、{sum(c["dense_complete"] for c in changed)} 条退步**。'
             '例如远程医疗 + 膀胱过度活动症的中英文组合问题（R285/R286）改善，但洪水成因 + HDL 的组合问题（R305/R306）退步。'
             '多问题共享有限候选名额时，词法信号也可能挤掉另一部分所需材料；[全部变化案例](retrieval/changed_cases.json)含目标来源与完整排名。','',
             '## 最终回答：覆盖、编造与信息不足','',
             f'在全部 **640 条测试查询**上固定 Top-3 和相同证据约束提示，对比纯向量与开发集选出的 **{labels[hybrid]}**。'
             '另外在新增的 160 条无答案 / 部分可回答问题上运行普通提示，比较明确核对证据边界的作用。每份回答独立真实生成，保留完整上下文和 API 记录。','',
             '| 检索与回答方式 | 回答数 | 模型评审完整正确 | 评审标记无依据事实 | 评审标记虚报查证 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for method,label in [('dense','纯向量 + 证据约束'),(hybrid,'混合检索 + 证据约束')]:
        m = answers['rag/'+method+'/all']
        text.append(f'| {label} | {m["n"]} | {fraction(m["complete_correct"],m["n"])} | {fraction(m["unsupported"],m["n"])} | {fraction(m["false_search"],m["n"])} |')
    text += ['', '在同一批 160 条困难问题、相同混合检索上下文上比较提示。两组问题本身都保留“仅根据指定摘要回答”的限制，普通提示也能看见这一限制：','',
             '| 回答提示 | 问题数 | 模型评审完整正确 | 评审标记无依据事实 | 评审标记虚报查证 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode,label in [('basic','普通提示'),('guarded','明确核对证据边界')]:
        m = paired['prompt_ablation'][mode]
        text.append(f'| {label} | {m["n"]} | {fraction(m["complete_correct"],m["n"])} | {fraction(m["unsupported"],m["n"])} | {fraction(m["false_search"],m["n"])} |')
    text += ['', '| 问题类型 | 数量 / 方法 | 纯向量完整正确 | 混合完整正确 | 混合明确说明缺失信息 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for kind,label in [('answerable','全部可回答'),('partial','部分可回答'),('unanswerable','资料不足 / 无答案'),('needs_history','缺少必要历史')]:
        a,b = [answers['rag/'+method+'/'+kind] for method in ['dense',hybrid]]
        text.append(f'| {label} | {a["n"]} | {fraction(a["complete_correct"],a["n"])} | {fraction(b["complete_correct"],b["n"])} | {fraction(b["acknowledges_limit"],b["n"])} |')
    text += ['', '“完整正确”同时要求回答有依据的部分、指出缺失信息，并且没有不受证据支持的事实或虚报查证。'
             '明确拒答不会自动判为通过：有足够材料却拒绝回答属于完整性失败。主题相关但没有具体答案的材料不能支持数字、效果或个人信息。'
             '声称查遍 PubMed、审批数据库等，但实际没有外部查询记录，会单独判为虚报查证。','',
             '评审按内容哈希打散并隐藏方法名；违规判断必须引用实际回答原文，引用 ID 另行校验。'
             '这些是模型辅助评审结果，保留每项理由供复核；不把检索命中率等同于医学回答准确率。','',
             f'程序检查发现纯向量 / 混合回答分别有 **{answers["rag/dense/all"]["invalid_citation"]} / {answers["rag/"+hybrid+"/all"]["invalid_citation"]} 份**使用了上下文中不存在的引用 ID。'
             '例如 [R163 的原始回答](answers/cases/R163__dense.json)使用了材料中没有的 `[MLP-123]`。这类错误即使事实看似合理也不能判为通过，因此不能声称模型完全不会编造引用。','',
             f'每个方法执行一次 temperature=0.2 的回答生成。640 对回答中，**{paired["identical_context_cases"]} 对输入上下文完全相同**，'
             f'其中 **{paired["identical_context_different_pass_flags"]} 对**仍出现通过 / 未通过差异，体现生成与评审波动。'
             '另附对相同上下文固定复用纯向量回答及评分的敏感性分析，保留两份原回答；不把这种随机差异归因于混合检索。'
             '[配对区间、同上下文对照和提示对照](answers/paired_comparison.json)。','',
             '### 评审可靠性复核','',
             '对全部初评未通过或引用无效的回答，以及固定种子抽取的每种方法 8 份通过回答，使用 GPT-5.5 / Codex CLI 独立复核。'
             '复核不提供原评分或方法名，明确区分实际召回材料与评审用标准答案。原始评分全部保留，不按复核后较高的成绩替换。'
             '这是失败样本富集的定向审计，不能将其通过率外推为全部回答的准确率。','',
             '| 复核范围 | 回答数 | 初评无依据事实标记 | 复核无依据事实标记 | 初评 / 复核虚报查证标记 | 完整性判断分歧 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,label in [('answers/all','RAG'),('memory_answers/all','本地 Mem0')]:
        m = audit[key]
        text.append(f'| {label} | {m["n"]} | {m["initial_unsupported"]} | {m["review_unsupported"]} | {m["initial_false_search"]} / {m["review_false_search"]} | {m["pass_disagreements"]} |')
    text += ['', '逐条检查评审理由可见，初评有把保守拒答当成事实编造、把标准答案引文误当作实际已召回材料的情况。'
             '因此上表的“评审标记”是待核查信号，不能直接称为已证实的幻觉率；引用 ID 不存在的错误另由程序确认。'
             'GPT-5.5 复核也属于模型辅助判断，没有替代专业人工评审。'
             '[全部抽样规则、原评与复核对照](judge_audit/reviews.json)、[两次判断的完整输入](judge_audit/selection.json)。','',
             '## 本地 Mem0：从历史召回到回答','',
             '当前产品使用 **Mem0 OSS 2.0.20 + 本地 Qdrant / SQLite**；事实抽取和 embedding 通过 API。'
             '本轮承接已完成的 120 个跨会话场景，使用其中 96 个测试场景关闭并重开本地存储后的真实 Top-10 召回（阈值 0.3），新增最终回答评测。'
             '这是对已有实际召回的回答延伸，没有把它写成重新执行了一次 Mem0 写入。','',
             '| 本地 Mem0 回答检查 | 数量 | 模型评审完整正确 | 评审标记无依据事实 | 明确说明缺失信息 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for kind,label in [('answerable','用户历史事实'),('partial','历史问题 + 未披露的发票编号')]:
        m = memory['memory/mem0_top10/'+kind]
        text.append(f'| {label} | {m["n"]} | {fraction(m["complete_correct"],m["n"])} | {fraction(m["unsupported"],m["n"])} | {fraction(m["acknowledges_limit"],m["n"])} |')
    text += ['', '复核发现旧记忆标签有 **11 项修正**：10 个时间线问题凭空指定 2022/2023 年，另一个问题的“未再复发”与第二次会话冲突。'
             '本轮依据原始用户陈述修正标签，并给评审原始会话及记录时间，防止把抽取错误当成真实用户事实。'
             '这是标注修正，不能当成 Mem0 能力提升；[修正前后及原话](../../medix-agent-swarm/evals/hybrid_grounding_v3/data/memory_label_corrections.json)完整保留，旧冻结文件未改写。','',
             '原始发票探针把历史问题与发票编号问题拼接在一起，因此本轮将全部 96 条统一归为部分可回答：要求回答已知历史，同时承认编号缺失。'
             '原始生成输入、分类修正前的评分和最终规则均保留。不能只说“不知道编号”就忽略前面的历史问题。','',
             '早期托管 Mem0 的接口失败与当前本地后端分开归档。本地测评历史中也出现过 HTTP 成功但正文为空、响应截断等 provider 故障；'
             '这些属于执行失败，和成功返回后的内容错误分别统计。[本地原始执行与异常记录](../component-benchmark-2026-09-08/memory/README.md)。','',
             '## 数据构建与复算','',
             '新增主题先按固定种子从未用作原有答案来源的摘要中抽取，再划分 20 个开发主题、80 个测试主题；每个主题包含英文问题、中文翻译、来源限定的缺细节问题、部分可回答问题。'
             '数据作者包含已完成的 Qwen3.5-27B 记录与 GPT-5.5 / Codex CLI，逐题注明来源；MiniMax M2.5 根据原文复核，程序检查证据引文。'
             '冻结前另有 10 项助手原文复核：9 项纠正翻译等价判断误读，1 项将非连续摘录改为单一问题及连续原文；全部原判定和修订理由保留。'
             '缺细节问题明确限定所指摘要，避免把“单篇摘要没写”错误标成“整个医学领域没有答案”。','',
             '- [完整协议与运行命令](../../medix-agent-swarm/evals/hybrid_grounding_v3/README.md)、[新增 400 条数据](../../medix-agent-swarm/evals/hybrid_grounding_v3/data/extension.jsonl)。',
             '- [全部 800 条排名](retrieval/rankings.json)、[参数选择](retrieval/selection.json)、[汇总](retrieval/summary.json)、[新增查询向量](retrieval/extension_vectors.npz)。',
             '- [全部检索改善与退步案例](retrieval/changed_cases.json)、[RAG 回答失败索引](answers/failures.json)、[Mem0 回答失败索引](memory_answers/failures.json)。',
             '- [RAG 回答与上下文](answers/cases/)、[逐项评审](answers/grades.json)、[汇总](answers/summary.json)。',
             '- [Mem0 回答与原始会话](memory_answers/cases/)、[逐项评审](memory_answers/grades.json)、[汇总](memory_answers/summary.json)。',
             '- [数据生成及复核记录](authoring/)、[发布文件校验清单](manifest.json)。','',
             f'本轮 API 响应中可核对的费用合计 **${cost["reported_cost_usd"]:.4f}**，包含生成、翻译、评分校准和格式修复；'
             '不包含沿用的旧向量与 Mem0 执行费用，也不包含 Codex 订阅额度。查询翻译增加一次模型调用，本实验以批次运行并缓存，未测单查询线上延迟。'
             '[按阶段费用与批次时间](cost_summary.json)。','',
             '这是固定英文摘要库上的中英文合成工程测评。统计单位、查询类型和后端均明确区分；正式产品中的 Milvus 检索路径与本轮本地快照检索适配器分别标注。', '']
    (RESULTS/'README.md').write_text('\n'.join(text),encoding='utf-8')
    methods = '''# 混合检索与最终回答测评运行说明

本套件延续 component_benchmark_v2 的冻结语料、原有查询和实际向量；新增 BM25 / 加权 RRF、查询翻译和最终回答评测。混合索引实现在 [hybrid_retrieval.py](../../knowledge/hybrid_retrieval.py)。它是本轮本地快照检索适配器，产品默认 Milvus 路径没有被悄然替换。

## 离线复算

从仓库根目录运行，需要 Python 3.11+、NumPy 与 httpx（只用于导入在线运行器，验证不调用网络）：

```powershell
python -m pip install numpy httpx pytest
python -m pytest medix-agent-swarm/evals/hybrid_grounding_v3/test_hybrid.py -q
python medix-agent-swarm/evals/hybrid_grounding_v3/verify.py
```

验证冻结输入、场景隔离、全部 800 条排名、开发集参数选择和回答分母；评分原文引文与汇总一并核对。模型评审本身不等于离线客观真值，原始回答和评审理由可单独审查。

## 在线运行

设置自己的 `OPENROUTER_API_KEY`。下列命令调用实际 embedding、查询翻译、回答和评分 API，会消耗额度；不要将密钥写进代码或提交。

```powershell
python medix-agent-swarm/evals/hybrid_grounding_v3/retrieval.py
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage generate --domain rag
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage grade --domain rag
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage generate --domain memory
python medix-agent-swarm/evals/hybrid_grounding_v3/answers.py --stage grade --domain memory
```

已完成结果按固定输入复用。要更换参数、模型或重新采样，应使用新的版本目录，不能覆盖这些冻结结果。数据构建使用 `prepare_codex.py --model gpt-5.5`，需要已登录 ChatGPT 的 Codex CLI；它只接收选定摘要生成 JSON，不访问检索结果。先前 API 作者的完成记录按逐源元数据保留。

RRF 网格与 tie-break 见检索协议；只有开发集决定参数。回答使用相同 Top-3 预算；普通提示对照仅用于新增的 160 条困难测试问题。Mem0 使用已保存的真实本地召回，不重新写入用户存储。

## 评分与历史记录

保留原始回答、提示、API 响应和逐项证据。格式错误只做有界修复，不因分数不理想重试。评分 v2 将不必要的拒答与肯定式事实编造分开；早期校准输出保留。记忆的 11 项标签修正见 `data/memory_label_corrections.json`，旧数据和旧分数不被覆盖。

`audit_judgments.py` 用 Codex CLI 独立复核全部初评失败和每种方法固定抽取的 8 份通过回答；结果是定向审计，不外推总体准确率，也不覆盖初评分数。CLI 遗漏项仅补该项；一处引文误字修正记录于 `judge_audit/quote_repairs.json`，保留原判定与原始输出。`answer_comparison.py` 生成配对区间与相同上下文敏感性分析，`case_studies.py` 导出全部失败索引，`costs.py` 汇总已有调用费用；`build_report.py` 从这些结果生成报告及哈希清单。

完整数据构成、数字和证据入口见[评测报告](../../../results/hybrid-grounding-2026-09-08/README.md)。
'''
    (HERE/'README.md').write_text(methods,encoding='utf-8')
    records = []
    for folder in [HERE,RESULTS]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='manifest.json':
                records.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size})
    runtime = ROOT/'medix-agent-swarm/knowledge/hybrid_retrieval.py'
    records.append({'path':runtime.relative_to(ROOT).as_posix(),'sha256':digest(runtime),'bytes':runtime.stat().st_size})
    write(RESULTS/'manifest.json',{'files':records,'counts':{'rag_queries':800,'rag_answers':1440,'memory_answers':288},
                                'source_attribution':'Source: MedlinePlus, National Library of Medicine'})
    print('Report and manifest generated')


if __name__=='__main__':
    main()
