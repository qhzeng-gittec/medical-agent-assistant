"""Render the completed T0-T3 development pilot without replacing holdout scores."""
import json
from pathlib import Path

from common.io import load_jsonl
from data_processing.prepare_format_factorial import DATA, ROOT


def table(headers, rows):
    return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in [headers,['---']*len(headers),*rows])


def main():
    result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    assert result['status']=='complete' and not result['same_final_score_conflicts']
    plan=result['plan']
    groups=result['groups']
    evaluation={r['source_id']:r for r in load_jsonl(DATA/'evaluation.jsonl')}
    questions={sid:json.loads((ROOT/'judge/questions'/f'{sid}.json').read_text(encoding='utf-8')) for sid in evaluation}
    knowledge=[q for sid,q in questions.items() if evaluation[sid]['task']=='knowledge']
    content={}
    for arm in ['T1','T2','T3']:
        content[arm+'-T0']={
            'same_semantic_answer':sum(q['models'][arm]['diagnosis']['semantic_answer_key']==q['models']['T0']['diagnosis']['semantic_answer_key'] for q in knowledge),
            **{key:result['contrasts']['knowledge/'+arm+'-T0']['answer_score'][key] for key in ['improved','worsened','unchanged']}}
    (ROOT/'output_content_analysis.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# T0–T3：最终答案形式与关系解释的开发集对照','',
           '本轮实际执行四组从 C 适配器开始的续训。原生思考解释与最终答案均参与训练；最终答案完整句比例和医学正确性分开评分。以下是开发集实验，不能替代 247 道留出测试的成绩，也不证明临床能力。','',
           '## 本轮结论','',
           '模型能学会完整句形式：T0 的完整句比例为 4.0%，T1 为 100%，T2 为 4.7%，T3 为 99.3%。因此，“连格式都没有学到”不符合本轮干预结果。原模型本来就能生成自然语言解释；知识最终答案常为短语，与实际训练目标的形式一致。要求自然语言不等于明确要求每个最终答案都写成主谓完整句。','',
           '只改答案形式的 T1 相对 T0，答案分仅 +0.67，95% 配对区间 [-3.00, +4.33]；解释分 -1.00。形式已经明显迁移，但没有可靠的语义收益证据。T1 有 120/150 道题的最终答案被判为与 T0 语义相同。','',
           '关系解释组 T2 的答案分 +4.00，区间 [-1.33, +9.33]，16 题提高、8 题降低；这是值得复验的线索。解释分却为 -0.67、联合分仅 +1.00，均不能确认改善。T3 的答案分 +2.00、解释分 -2.33、联合分 -1.00，也没有两项改动叠加获益的证据。答案交互效应为 -2.67，区间 [-8.00, +2.67]，不能认定存在负交互。','',
           '四组续训模型的 VQA 答案分均为 68.75，起点 C 为 81.25：同样的两题从满分变成零分。仅 16 张图，区间宽，不能宣称已证明整体视觉能力下降，但它是必须进一步检查的回归线索；不能只看知识分最高就选 T2。','',
           '## 实验范围','',
           f"每组 {plan['accepted_knowledge']} 道知识题 + 192 道相同 VQA 回放，两个 epoch，{plan['optimizer_steps']} 个优化步；batch=1、梯度累积=4、学习率 2e-5、seed=42。均加载同一个 C 适配器和新优化器。图像上限 512，视觉编码器冻结。",'',
           'T0：当前最终答案与当前解释；T1：同义完整句答案与当前解释；T2：当前最终答案与关系解释；T3：同义完整句答案与关系解释。C_start 为未续训起点。T2 复用已有逐题生成、审阅通过的关系解释，不能说每条原解释都有缺陷。','',
           '知识评测为项目 validation 中全部 150 道题；VQA 为该开发集 16 张独立图像各一道题。开发集与本轮训练按样本和来源分组隔离，但它曾用于此前开发实验，不能宣称全新测试。项目现有 test 没有用于选择本轮训练样本或评分。VQA 的 train-/test- 原始 ID 前缀不代表项目重新按图像划分后的 split，隔离依据实际配方和 source_group。','',
           '同一任务各组使用相同问题、greedy、总预算 2048、思考预算 1792。gpt-5.6-sol 匿名评分，答案与解释各 0/1/2；归一化分为均值乘 50，联合分取两项原始分的最小值。语义等价答案必须同分，完整句比例独立于正确性。','',
           '## 知识结果','']
    lines.append(table(['组别','有效题数','答案分','解释分','联合分','完整句比例','答案中位词数','解释中位词数'],[
        [a,groups['knowledge/'+a]['n'],*[f"{groups['knowledge/'+a][m]:.2f}" for m in ['answer_score','reasoning_score','joint_score','full_sentence_percent']],
         groups['knowledge/'+a]['median_answer_words'],groups['knowledge/'+a]['median_reasoning_words']] for a in ['C_start','T0','T1','T2','T3']]))
    lines+=['','## VQA 回归检查','',table(['组别','有效题数','答案分','解释分','联合分'],[
        [a,groups['vqa/'+a]['n'],*[f"{groups['vqa/'+a][m]:.2f}" for m in ['answer_score','reasoning_score','joint_score']]] for a in ['C_start','T0','T1','T2','T3']]),'',
        'VQA 仅 16 张图，只能作小规模回归线索，不能据此证明视觉能力完全保持。','',
        '## 预定配对与交互作用','']
    comparisons=[]
    for name in ['T1-T0','T2-T0','T3-T0','T3-T2','T3-T1','T0-C_start','form','relation','interaction']:
        cells=[name]
        for metric in ['answer_score','reasoning_score','complete_sentence']:
            value=result['contrasts']['knowledge/'+name][metric]
            cells.append(f"{value['delta']:+.2f} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}]")
        comparisons.append(cells)
    lines += [table(['对照','答案分差 [95% CI]','解释分差 [95% CI]','完整句百分点差 [95% CI]'],comparisons),'',
              'form=(T1+T3−T0−T2)/2；relation=(T2+T3−T0−T1)/2；interaction=T3−T2−T1+T0。区间为 20,000 次按学科分层、按题配对 bootstrap 的名义区间，未作多重比较校正，不包含训练种子和评分器波动。','',
              '## 如何解读','',
              '完整句比例上升只证明输出形式向目标迁移；只有自由回答语义分提高，才能讨论任务收益。若形式变化而语义区间跨零，不能将其称为格式学习失败，也不能说性能已经提升。若 T0 相对 C_start 也退步，应考虑续训本身的影响。','',
              'T2 同时改变关系表述、解释长度及监督 token 数，不能将它的变化全部归因于新增知识。四组样本和步数一致，但 token 预算并不相同。不同目标上的训练 loss 不可直接横向排序。','',
              '这是单种子的小规模续训对照，不是全量从基座重训。候选方案应再用配对训练种子和未参与选择的数据确认；不根据当前开发集分数替换原留出测试成绩。','',
              f"本轮不可判分题：{len(result['invalid_questions'])}；相同文本或语义组评分冲突：{len(result['same_final_score_conflicts'])}。训练配方哈希、起点哈希与四组字段不变量已检查。",'',
              'T2 复用的已审核关系池中，一部分补齐关键连接，另一部分主要展开已有关系；本轮并不是每条样本都加入了新的易混概念对比。','',
              '## 核心答案的逐题变化','',
              table(['对照','语义相同 / 150','答案分提高','答案分降低','答案分相同'],[
                  [name,value['same_semantic_answer'],value['improved'],value['worsened'],value['unchanged']] for name,value in content.items()]),'',
              '“同分”不等于“同义”：改成另一种错误答案也可能同为零分。语义分组来自匿名评分器，只作为输出内容的辅助分析。','',
              '## 直接阅读输出','',
              '以下样例展示具体变化；整体结论依据全部 150 道知识题，不依据个别案例。正确性描述以本轮参考与逐题评分为依据。','']
    for sid,title,interpretation in [
        ('medmcqa-002bc215-c803-46d4-bb72-ac7c5a31563a','形式学会，错误保留',
         '题目问 Touton 巨细胞的典型相关病变。四组都仍回答 tuberculosis，T1/T3 只是改成完整句；本轮参考为黄色瘤等脂质性病变。这里已经学会表达形式，却没有纠正概念关联。'),
        ('medmcqa-1efc680b-ed5d-4c2c-8049-e3defda927c4','关系组改对的一题',
         '题目问哪种试剂不是标准革兰染色的组成。T0/T1 选 ethanol，T2/T3 改为参考答案 methylene blue，解释也正确识别了 ethanol 的脱色剂作用。'),
        ('medmcqa-1ecf1005-d814-4f43-a30a-e6a0f61cf0bf','两种改动并不保证叠加',
         '题目问精原细胞瘤的卵巢对应肿瘤。T1/T2 答对，T3 却答成颗粒细胞瘤；其完整句和连贯解释仍包含错误关系。这个例子不能单独证明负交互，但说明长句与长解释不能充当正确性的代理指标。')]:
        q=questions[sid]
        lines += [f'### {title}','',interpretation,'',
                  table(['组别','最终答案原文','答案 / 解释原始分'],[
                      [arm,q['models'][arm]['final_answer'],f"{q['models'][arm]['diagnosis']['answer_score']} / {q['models'][arm]['diagnosis']['reasoning_score']}"] for arm in ['T0','T1','T2','T3']]),'',
                  f"[完整解释与匿名评分](<{(ROOT/'judge/questions'/f'{sid}.json').as_posix()}>)",'']
    lines += ['## 对低 loss 和下一步实验的判断','',
              '本轮支持“输出形式随监督改变，但知识正确性没有同步稳定提高”，不支持“模型不会自然语言”或“错误全是短答案过拟合造成的”。自然语言解释、最终答案形式、事实关联、解释到答案的一致性是不同的检查项。不能仅凭输出变短或训练 loss 低，就给出过拟合形式的因果结论。','',
              '本轮没有新增分段 loss 实验。此前的正确解释前缀 / 自生成解释前缀探测见下方链接；它说明 loss 的条件需要区分，但仍不能把全部性能差异归因于单一机制。','',
              '下一步优先顺序：','',
              '1. 先用 T0/T2 做至少三个配对训练种子的复验，保持本轮初始化起点、样本、步数和评测方式一致，检查 +4 分线索是否稳定，并同时检查解释和 VQA。不要先把 T2 的开发集最高分写成已验证提升。',
              '2. 将关系改写进一步收窄为限定条件和易混概念的成对训练：改变给药途径、类型、比较对象或因果方向时，答案必须随之改变。与同长度的普通释义解释对照，尽量匹配监督 token；按来源及关系族隔离，并分别报告同一关系的新问法和未训练关系族。这样更直接检验关键关系绑定，而不是笼统延长解释。',
              '3. 单独检查续训带来的图像回归：在更大的按图像隔离开发集上，以 T0 为对照，一次只改变学习率或 VQA 回放比例，同时考察知识和图像输出。当前两道回归题只能用于诊断，不能靠记住它们来证明视觉能力恢复。','',
              'T1/T3 可以作为“完整句形式能否学习”的证据；若产品要求完整句，可保留这种监督形式，但本轮没有证明它能提高知识正确性。FFN 没有在本轮变化，因此不能用本轮结果宣称已定位 FFN 有无收益的原因。','',
              '166/166 题有评分，830/830 份输出完整，无空最终答案或长度截断。评分返回中两处标签缺少对应引用，一处样本 ID 漏写一个字符。原始返回已保留；按引用列表清理标签，并以全部候选回答哈希确认 ID 后修复。没有改动数值评分；缓存重验与全量汇总通过。','',
              '## 原始证据','']
    for label,path in [('固定方案',DATA/'experiment_plan.json'),('字段与监督检查',ROOT/'data_verification.json'),
                       ('逐组指标与全部区间',ROOT/'results.json'),('匿名逐题评分',ROOT/'judge/questions'),
                       ('逐组生成',ROOT/'generation'),
                       ('输出完整性与隔离核查',ROOT/'execution_verification.json'),
                       ('语义内容变化',ROOT/'output_content_analysis.json'),
                       ('评分原始返回',ROOT/'judge/original_calls'),
                       ('标签修正记录',ROOT/'judge/flag_normalization.json'),
                       ('样本 ID 修正记录',ROOT/'judge/id_normalization.json'),
                       ('此前的短答案与条件损失诊断',ROOT.parent/'independent_holdout_20260906/output_style/短回答形式与条件损失诊断.md')]:
        lines.append(f'- [{label}](<{path.as_posix()}>)')
    (ROOT/'T0-T3实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote T0-T3 development report.')


if __name__=='__main__':
    main()
