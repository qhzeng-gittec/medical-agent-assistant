"""Summarize every frozen run; repeated runs remain clustered by scenario."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).parent/'results/architecture_compare_20260908_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(rows,key):
    return statistics.mean(r[key] for r in rows)


def concurrency(trace):
    events=[]
    for r in trace:
        if r['event']=='worker_span':
            events.extend([(r['started'],1),(r['ended'],-1)])
    active=maximum=0
    for _,change in sorted(events):
        active+=change
        maximum=max(maximum,active)
    return maximum


rows=[]
for path in sorted((ROOT/'runs').glob('*/result.json')):
    row=read(path)
    gradepath=ROOT/'grades'/row['run_id']/'result.json'
    row['grade']=read(gradepath) if gradepath.exists() else None
    row['max_worker_concurrency']=concurrency(read(path.parent/'trace.json'))
    rows.append(row)
scheduler=[]
for path in sorted((ROOT/'scheduler').glob('*/result.json')):
    row=read(path)
    row['max_worker_concurrency']=concurrency(read(path.parent/'trace.json'))
    scheduler.append(row)
assert len(rows)==32 and all(r['status']=='completed' for r in rows)
assert len(scheduler)==6 and all(r['status']=='completed' for r in scheduler)
assert all(len(r['records'])==3 and all(x['success'] for x in r['records']) for r in scheduler)

groups={}
for arm in ['single','swarm']:
    subset=[r for r in rows if r['arm']==arm]
    graded=[r for r in subset if r['grade']]
    groups[arm]={'runs':len(subset),'scenarios':len({r['case_id'] for r in subset}),
        'mean_case_seconds':mean(subset,'seconds'),'median_case_seconds':statistics.median(r['seconds'] for r in subset),
        'automatic_grades_completed':len(graded),
        'llm_requests':sum(r['metrics']['llm_requests'] for r in subset),
        'cost_usd':sum(r['metrics']['cost_usd'] for r in subset),
        'prompt_tokens':sum(r['metrics']['prompt_tokens'] for r in subset),
        'completion_tokens':sum(r['metrics']['completion_tokens'] for r in subset),
        'actual_multi_worker_batches':sum(r['metrics']['multi_worker_batches'] for r in subset),
        'provider_errors':sum(r['metrics']['provider_errors'] for r in subset),
        'max_observed_worker_concurrency':max(r['max_worker_concurrency'] for r in subset)}

bycase=[]
for case in sorted({r['case_id'] for r in rows}):
    entry={'case':case}
    for arm in ['single','swarm']:
        selected=[r for r in rows if r['case_id']==case and r['arm']==arm]
        entry[arm]={'seconds':mean(selected,'seconds'),
                    'calls':sum(r['metrics']['llm_requests'] for r in selected)/len(selected)}
    bycase.append(entry)
quality={'automatic_grades_completed':sum(bool(r['grade']) for r in rows),
         'planned_grades':len(rows),'status':'incomplete_http_402',
         'accuracy_improvement_claim_supported':False,
         'local_review':'Qualitative review only; original automatic grades preserved. See local_review_notes.json.'}

sg={}
for arm in ['serial','swarm']:
    selected=[r for r in scheduler if r['arm']==arm]
    sg[arm]={'runs':len(selected),'mean_seconds':mean(selected,'seconds'),
             'seconds':[r['seconds'] for r in selected],
             'llm_requests':sum(r['metrics']['llm_requests'] for r in selected),
             'completion_tokens':sum(r['metrics']['completion_tokens'] for r in selected),
             'max_worker_concurrency':max(r['max_worker_concurrency'] for r in selected)}
sg['latency_reduction_percent']=(1-sg['swarm']['mean_seconds']/sg['serial']['mean_seconds'])*100
sg['speedup_ratio']=sg['serial']['mean_seconds']/sg['swarm']['mean_seconds']
total=0
for ledger in ROOT.rglob('api_ledger.jsonl'):
    latest={}
    for line in ledger.read_text(encoding='utf-8').splitlines():
        item=json.loads(line);latest[item['request_id']]=item
    total+=sum(item['charged_or_reserved_usd'] for item in latest.values())
affected_cases={r['case_id'] for r in rows if r['metrics']['provider_errors']}
network_sensitivity={arm:statistics.mean(r['seconds'] for r in rows if r['arm']==arm and r['case_id'] not in affected_cases)
                     for arm in ['single','swarm']}
result={'groups':groups,'quality':quality,'bycase':bycase,'scheduler':sg,'total_cost_usd':total,
        'network_affected_cases':sorted(affected_cases),'mean_latency_excluding_entire_network_affected_case':network_sensitivity}
(ROOT/'comparison_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['# 单 Agent 与当前多 Agent 架构实测','',
       '模型：MiniMax M2.5；8 个开发场景，每组重复 2 次，共 32 组案例执行，全部获得最终回答。Qwen3.5-27B 匿名逐项评审完成 17/32 组后因 HTTP 402 余额不足中断，没有继续付费请求。',
       '测试使用真实模型 API、真实查询 embedding、既有 15 文档本地向量索引；没有接入微调医疗模型、在线 Mem0 或外部搜索。',
       '预跑发现对照脚本工具引用 ID 缺失，修复后通过 2 项回归检查，重新冻结并重跑全部正式组；预跑保存在旧目录，不计入成绩。','',
       '## 1. 单 Agent 与多 Agent','',
       '| 指标 | 单 Agent | 当前多 Agent |','|---|---:|---:|']
for label,key,fmt in [('平均案例耗时（秒）','mean_case_seconds','.2f'),('案例耗时中位数（秒）','median_case_seconds','.2f'),
                      ('已完成自动评分的执行数 / 16','automatic_grades_completed','d'),
                      ('模型请求数','llm_requests','d'),('供应商连接或请求错误数','provider_errors','d'),('模型与检索费用（美元）','cost_usd','.4f')]:
    lines.append(f'| {label} | {groups["single"][key]:{fmt}} | {groups["swarm"][key]:{fmt}} |')
lines+=['',f'当前多 Agent 平均案例耗时为单 Agent 的 {groups["swarm"]["mean_case_seconds"]/groups["single"]["mean_case_seconds"]:.2f} 倍。本次没有测出完整案例提速。',
        '质量评分未完成，且本地复核发现评分器把 A03 要求中的“NHS 或 CDC”误读为必须提供 CDC 链接，误判了两次多 Agent 回答。保留原始评分，另列复核记录；不将缺失项算作失败，也不拼接自动和本地评分计算准确率提升。',
        '本地复核还发现 A04 多 Agent 回答未满足指定 NHS 来源要求，A01_swarm_r2 未回应无症状能否排除风险。当前证据不足以支持准确率提升的简历表述。评审只检查开发场景的任务要求，不等同于临床准确率。','',
        f'服务错误涉及场景 {sorted(affected_cases)}。主表保留这些运行；若整场景的两种架构及两次重复均排除，平均案例耗时单 Agent {network_sensitivity["single"]:.2f}s、多 Agent {network_sensitivity["swarm"]:.2f}s。此为敏感性分析，不替代主结果。','',
        '| 场景 | 单 Agent 均值秒 | 多 Agent 均值秒 | 单 Agent 平均模型请求 | 多 Agent 平均模型请求 |',
        '|---|---:|---:|---:|---:|']
for c in bycase:
    lines.append(f'| {c["case"]} | {c["single"]["seconds"]:.2f} | {c["swarm"]["seconds"]:.2f} | {c["single"]["calls"]:.1f} | {c["swarm"]["calls"]:.1f} |')
lines+=['','## 2. 固定子任务串行与并行','',
        '相同三个任务分别由诊断、咨询、研究 Worker 执行；使用真实模型和工具，各模式 3 次，交替顺序，期间不运行其他本次评测任务。',
        '该测试只切换 Worker 并发，排除总控规划及最终汇总；不能把子任务阶段的比例套用到完整问诊。三次执行的输出长度和云端负载仍会波动。','',
        '| 指标 | 串行 | 并行 |','|---|---:|---:|',
        f'| 平均耗时（秒） | {sg["serial"]["mean_seconds"]:.2f} | {sg["swarm"]["mean_seconds"]:.2f} |',
        f'| 三次耗时（秒） | {[round(x,2) for x in sg["serial"]["seconds"]]} | {[round(x,2) for x in sg["swarm"]["seconds"]]} |',
        f'| 模型请求数 | {sg["serial"]["llm_requests"]} | {sg["swarm"]["llm_requests"]} |',
        f'| 输出 Token | {sg["serial"]["completion_tokens"]} | {sg["swarm"]["completion_tokens"]} |',
        f'| 观测最大并发 Worker | {sg["serial"]["max_worker_concurrency"]} | {sg["swarm"]["max_worker_concurrency"]} |','',
        f'子任务阶段平均耗时缩短 {sg["latency_reduction_percent"]:.1f}%，加速 {sg["speedup_ratio"]:.2f} 倍。',
        f'自然任务中，多 Agent 共出现 {groups["swarm"]["actual_multi_worker_batches"]} 个多 Worker 同轮批次，最大实际 Worker 并发为 {groups["swarm"]["max_observed_worker_concurrency"]}。','',
        'A03_swarm_r1 出现 120 秒 Worker 超时，后续总控继续分派；多次 Worker 输出原始工具调用文本，导致重复检索和额外模型轮次。完整案例计时包含这些实际开销。固定子任务测试说明并发机制有速度收益，尚不能说明当前完整架构更快或更准。','',
        '## 3. 对照公平性与复核入口','',
        '单 Agent 复用当前产品的档案、上下文和安全处理，直接持有全部 7 个底层工具，允许并行调用工具；最多 10 轮模型迭代、24 次底层工具调用。多 Agent 保持当前产品的 4 轮总控、每次 Worker 最多 2 次工具调用。没有刻意剥夺单 Agent 的工具或记忆能力。',
        '角色提示词、上下文隔离与可用总推理预算属于架构整体差异，因此自然任务对照测的是这套具体实现，不是所有单/多 Agent 方案的优劣。每个执行使用隔离进程，模型与温度相同；先后顺序平衡；同时最多运行两组案例。',
        f'本轮计费与保守预留合计约 ${total:.4f}，账本保留原始供应商费用。','',
        '原始数据：`runs/*/result.json` 与 `trace.json`；匿名评分：`grades/*/result.json`；串并行：`scheduler/*`；冻结配置：`protocol.json`；机器可读汇总：`comparison_summary.json`。']
(ROOT/'架构对照实测结果.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
