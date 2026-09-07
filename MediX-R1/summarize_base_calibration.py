import json
import statistics
from pathlib import Path

from evaluate_base_calibration import ROOT, OLD, TASKS, read_rows, save

BOUNDED_ROOT = ROOT.parent / 'base_calibration_bounded_v2'


def metrics(rows):
    return dict(n=len(rows), score=50*statistics.mean(r['judge']['score'] for r in rows),
        mean_tokens=statistics.mean(r['generated_tokens'] for r in rows),
        length_stops=sum(r['stop_reason']=='length' for r in rows),
        eos_stops=sum(r['stop_reason']=='eos' for r in rows),
        repetition_stops=sum(r['stop_reason']=='repetition' for r in rows),
        overlong_stops=sum(r['stop_reason']=='overlong_suspected_repetition' for r in rows),
        forced_thinking=sum(r['forced_reasoning_end'] for r in rows),
        empty_final=sum(not r['final_answer'].strip() for r in rows))


def main():
    old = {t:read_rows(OLD/f'{t}.jsonl') for t in TASKS}
    data = {'original_greedy_2k': {t:metrics(rows) for t,rows in old.items()}}
    old_all = [r for rows in old.values() for r in rows]
    data['original_strata'] = {label:metrics([r for r in old_all if (r['stop_reason']=='length')==truncated])
        for label,truncated in [('truncated',True),('not_truncated',False)]}
    data['original_greedy_2k']['overall'] = metrics(old_all)
    for directory in [*ROOT.iterdir(), *BOUNDED_ROOT.iterdir()]:
        summary = directory/'validation/summary.json'
        if not summary.exists():
            continue
        candidate = {}
        pairs = []
        for t in TASKS:
            rows = read_rows(directory/'validation'/f'{t}.jsonl')
            baseline = {r['source_id']:r for r in old[t]}
            assert {r['source_id'] for r in rows} == set(baseline)
            candidate[t] = metrics(rows)
            for r in rows:
                b=baseline[r['source_id']]
                pairs.append(dict(task=t,id=r['source_id'],old_score=b['judge']['score'],new_score=r['judge']['score'],
                    old_truncated=b['stop_reason']=='length',new_truncated=r['stop_reason']=='length',
                    new_stop_reason=r['stop_reason'],
                    old_tokens=b['generated_tokens'],new_tokens=r['generated_tokens'],
                    old_answer=b['final_answer'],new_answer=r['final_answer'],new_judge=r['judge']['explanation']))
        all_rows=[r for t in TASKS for r in read_rows(directory/'validation'/f'{t}.jsonl')]
        candidate['overall']=metrics(all_rows)
        candidate['paired']={label:sum((r['new_score']-r['old_score']>0 if label=='wins' else r['new_score']-r['old_score']<0 if label=='losses' else r['new_score']==r['old_score']) for r in pairs) for label in ('wins','ties','losses')}
        candidate['old_truncated_subset']=dict(n=28,old_score=50*statistics.mean(r['old_score'] for r in pairs if r['old_truncated']),
            new_score=50*statistics.mean(r['new_score'] for r in pairs if r['old_truncated']))
        candidate['pairs']=pairs
        data[directory.name]=candidate
    save(BOUNDED_ROOT/'comparison.json',data)
    columns=['设置','VQA','知识','病例','上下文','综合分','平均 token','length 截断','重复停止','过长停止','空最终答案']
    lines=['# 基座能力补测', '', '固定原验证集四任务各 20 条，采用原判分标准。所有新设置使用官方建议的分任务采样参数；原结果是 greedy。分数同时评估结论与推理，不是纯答案准确率。', '',
        '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for name,d in data.items():
        if name=='original_strata':continue
        overall=d['overall']
        values=[name,*[f'{d[t]["score"]:.1f}' for t in TASKS],f'{overall["score"]:.3f}',f'{overall["mean_tokens"]:.1f}',str(overall['length_stops']),str(overall['repetition_stops']),str(overall['overlong_stops']),str(overall['empty_final'])]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', '## 解释边界', '',
        '- 原始 length 截断组 28 条，平均 37.5 分；未 length 截断组 52 条，平均约 64.42 分。两组题目不同，难度也可能不同，不能把差值当成截断的因果损失。',
        '- 原 32K 思考组已按用户要求停止，只有 19 条 VQA 原始输出，不参与完整组评分。新组在独立目录对全部 80 题统一使用 6144 token 上限及重复停止规则，没有选择性重跑失败题。',
        '- 新组从 512 token 起每 64 token 检查最近 2048 token；空白规范化后同一行至少 50 字符且出现 6 次即停止。达到 6144 上限标为过长疑似重复，不能据此认定实际重复。自然结束、确认重复和过长停止分开计数，所有失败保留并按原标准评分。',
        '- 2K 思考组强制结束思考并预留 256 最终 token，新组不强制闭合思考。预算与停止策略同时变化，不是严格单变量的长度因果对照。阈值依据同一验证集的已有输出选取，属于探索性测量。',
        '- 非思考组通过聊天模板关闭思考，整个输出作为最终回答交给相同 judge；因此它衡量另一种推理配置下的可用能力。',
        '- 每种采样配置目前只有一个固定随机种子。不能把最高一次得分称为真实能力上限，也不能把新基座分数与旧解码下的适配器分数直接视为公平排名。',
        '- 80 条验证样本只能衡量该样本集合；没有开展独立 test 或临床验证。',
        '', '官方参数来源：https://huggingface.co/Qwen/Qwen3.5-2B',
        '', '完整逐题分差、旧截断子集变化见 comparison.json；每种设置的原始输出、judge 和 summary 在对应 validation 文件夹。', '']
    (BOUNDED_ROOT/'calibration_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k: {kk:vv for kk,vv in v.items() if kk!='pairs'} for k,v in data.items()},ensure_ascii=False))


if __name__=='__main__':
    main()
