"""Promote the completed benchmark; retain historical result files and annotations."""
from common import BASE, RESULTS, ROOT, digest, read, write


def main():
    assert read(RESULTS/'answers/execution_status.json')['completed']
    s = read(RESULTS/'answers/summary.json')
    m = read(RESULTS/'memory_answers/summary.json')
    hybrid = read(RESULTS/'answers/protocol.json')['selected_method']
    a,b = s['rag/dense/all'],s['rag/'+hybrid+'/all']
    memory_a,memory_b = m['memory/mem0_top10/answerable'],m['memory/mem0_top10/partial']
    path = ROOT/'README.md'
    original = path.read_text(encoding='utf-8')
    lines = original.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('更新于 **2026-09-08**：'):
            lines[i] = '更新于 **2026-09-08**：新增 **800 条查询 / 1,016 篇语料**的 BM25、向量与混合检索测评，包含 **1,440 份 RAG 回答、288 份本地 Mem0 历史回答**及逐项证据检查；公开数据、开发集调参记录、实际向量与离线复算。'
        elif line.startswith('| RAG 保留多个证据候选 |') or line.startswith('| 多语言混合检索实验 |'):
            lines[i] = '| 多语言混合检索实验 | 纯向量 → 查询翻译 + BM25/向量 RRF，Top-3 完整来源覆盖 **408/432 → 410/432**；Top-5 **421/432 → 424/432** | 800 查询、1,016 篇语料；公开快照适配器，参数只在开发集选择；Top-3 差值区间包含零，收益较小 |'
        elif line.startswith('| Mem0 跨会话事实检索 |') or line.startswith('| 本地 Mem0 历史问答 |'):
            lines[i] = f'| 本地 Mem0 历史问答 | 历史问题完整正确 **{memory_a["complete_correct"]}/192**；历史 + 缺失信息组合问题 **{memory_b["complete_correct"]}/96** | 96 个测试用户，使用实际重开后的本地召回；根据原始会话核对更正、人物和时间；旧标签问题单独修正 |'
        elif line.startswith('**RAG 与 Mem0 已提供完整独立测评：**') or line.startswith('**检索与最终回答分别测量：**'):
            lines[i] = f'**检索与最终回答分别测量：** [混合检索、证据充分性与本地 Mem0 完整报告](results/hybrid-grounding-2026-09-08/README.md)。在同一批 640 条测试查询上，纯向量 / 混合检索的模型评审完整正确回答为 **{a["complete_correct"]}/640、{b["complete_correct"]}/640**；另在相同的 160 条困难问题上比较普通提示与证据约束提示。报告区分无依据事实、虚报查证、有信息却拒答，并公开评审分歧与生成波动的同上下文对照。混合检索为本轮可运行的快照实验，产品默认检索仍为 Milvus 路径。'
        elif line.startswith('[RAG 报告与实际向量]') or line.startswith('[完整混合检索与回答报告]'):
            lines[i] = '[完整混合检索与回答报告](results/hybrid-grounding-2026-09-08/README.md) · [新数据与复算命令](medix-agent-swarm/evals/hybrid_grounding_v3/README.md) · [原始 400 查询与 Mem0 执行记录](results/component-benchmark-2026-09-08/README.md)'
    text = '\n'.join(lines)+'\n'
    text = text.replace('[RAG / Mem0 独立测评](results/component-benchmark-2026-09-08/README.md)',
                        '[混合检索与回答测评](results/hybrid-grounding-2026-09-08/README.md)')
    assert text!=original or '800 条查询' in text
    path.write_text(text,encoding='utf-8')
    index = ROOT/'results/README.md'
    index_text = index.read_text(encoding='utf-8')
    entry = '新增 [800 查询的混合检索与最终回答测评](hybrid-grounding-2026-09-08/README.md)：1,016 篇资料、1,440 份 RAG 回答、288 份本地 Mem0 回答；包含 BM25 / 向量 / RRF 消融、证据不足检查、评审复核和离线复算。'
    if entry not in index_text:
        first,rest = index_text.split('\n',1)
        index.write_text(first+'\n\n'+entry+'\n'+rest,encoding='utf-8')
    notices = {
        ROOT/'results/rag-2026-09-08/README.md':'本页保留早期 72 条开发回归。最新 [800 条查询、混合检索与最终回答完整测评](../hybrid-grounding-2026-09-08/README.md)已扩展到 1,016 篇语料，包含 BM25 / 向量 / RRF、证据不足与本地 Mem0 回答检查。',
        BASE/'README.md':'最新 [800 条查询、混合检索与最终回答测评](../hybrid-grounding-2026-09-08/README.md)承接本页冻结数据，增加新主题和回答层检查。本页保留原始执行记录；Mem0 旧标签问题的修正见新报告。',
        BASE/'rag/README.md':'最新 [800 条查询的 BM25 / 向量 / 混合检索及回答测评](../../hybrid-grounding-2026-09-08/README.md)已发布。本页继续保存原有 400 查询的独立结果与向量。',
        BASE/'memory/README.md':'最新 [本地 Mem0 最终回答与标签复核](../../hybrid-grounding-2026-09-08/README.md#本地-mem0从历史召回到回答)发现 11 项旧标准答案问题，并补充 288 份实际回答。本页 176/192、184/192 是原冻结标签下的历史计数，不应将标注修正视为记忆能力提升。',
        ROOT/'results/2026-09-08/README.md':'新增 [800 查询的混合检索与最终回答测评](../hybrid-grounding-2026-09-08/README.md)：1,016 篇语料、1,440 份 RAG 回答、288 份本地 Mem0 回答；提供开发集选参、证据不足检查、配对分析和离线复算。'}
    changed = []
    for target,notice in notices.items():
        newline = '\r\n' if b'\r\n' in target.read_bytes() else '\n'
        text = target.read_text(encoding='utf-8')
        if notice not in text:
            first,rest = text.split('\n',1)
            target.write_text(first+'\n\n> '+notice+'\n'+rest,encoding='utf-8',newline=newline)
        changed.append(target)
    # Only documentation entries change; frozen data and runtime hashes stay intact.
    for manifest,base in [(BASE/'manifest.json',ROOT),(ROOT/'results/rag-2026-09-08/manifest.json',ROOT/'results/rag-2026-09-08')]:
        value = read(manifest)
        for record in value['files']:
            target = base/record['path']
            if target in changed:
                record.update(sha256=digest(target),bytes=target.stat().st_size)
        write(manifest,value)
    print('Updated project homepage and historical report navigation')


if __name__=='__main__':
    main()
