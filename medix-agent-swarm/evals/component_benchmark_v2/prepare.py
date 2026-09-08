"""Build source-grounded RAG probes and fictional memory scenarios; freeze before retrieval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re

from api import API, read, write

HERE = Path(__file__).resolve().parent


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(html):
    parser = Text()
    parser.feed(html)
    return ' '.join(' '.join(parser.parts).split())


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


RAG_INSTRUCTION = '''Create source-grounded retrieval evaluation questions. Return JSON {"items":[...]}.
For EACH supplied source return {"id":same id,"en":English patient question,"zh":Chinese patient question,
"evidence_id":the exact id string of ONE supplied evidence_options entry supporting BOTH questions}.
Ask one concrete question about a supported detail (cause, symptom, prevention, evaluation, management).
The two questions express the SAME information need, not two facts. Use ordinary patient language.
Include enough topic context to identify the condition but do not copy a source sentence into the question.
Select an evidence option and ask about a concrete detail stated in that option. Do not ask for a personalized dose or individual diagnosis. Do not invent content. No answers in questions.
These are synthetic topic-source retrieval labels, not a clinician-authored benchmark.'''

MEMORY_CATEGORIES = [
    '工作作息与睡眠时间变化', '运动限制与日常活动', '药物过敏及具体反应',
    '用户和家人的经历区分', '确诊与待检查的不确定性区分', '按医嘱停用药物的时间线',
    '预约复查与检查准备安排', '搬家旅行与生活环境变化', '用户更正先前陈述',
    '饮食偏好与个人生活习惯', '多项历史事实的跨会话汇总', '否定事实与助手建议区分',
]
MEMORY_INSTRUCTION = '''编写虚构患者记忆检索测试，不是医学建议。返回 JSON {"scenarios":[...]}，共10个独立场景。
每场景字段: id(按给定前缀01至10), category(给定类别), turns(恰好2个对象，字段user和assistant),
queries(恰好2个对象，字段id,query,expected_facts,required_groups)。
每条user为50到110个汉字，两个不同会话，第二次补充或更正；assistant只简短确认，不给新医疗事实。
两个query问已披露的具体个人历史，使用自然改写而非照抄。
expected_facts是完整中文事实句数组(1至3条)，明确是谁、当前或既往状态及用户提供的时间。
required_groups是关键词组的数组，每组为可接受同义短语数组，所有组均出现才算词汇覆盖。
关键词只能来自事实，不能包含医疗建议，避免把泛化词作为完整事实判定。
10个故事需在生活背景、事件、实体、关系、时间线、问法上有实质差异，不要仅替换姓名年龄数字。
均为成年人，不含真实身份、联系方式、身份证或地址。不要凭空把未确诊变成确诊。
对停药与更正，历史记录可以存在，query与expected_facts须明确时间归属。
合成身份按id隔离；后续另加未知用户隔离查询。'''


def main(args):
    target = HERE / 'data'
    target.mkdir(exist_ok=True)
    if (target / 'freeze.json').exists():
        raise RuntimeError('Dataset already frozen; use a new benchmark version')
    api = API(args.work / 'generation', max_requests=120)
    topics = read(args.topics)
    corpus = [{'id': 'MLP-' + t['id'], 'title': t['title'], 'url': t['url'],
               'content': plain(t['summary_html'] or ''), 'groups': t['groups'],
               'source': 'MedlinePlus, National Library of Medicine', 'language': 'en'} for t in topics]
    corpus = [d for d in corpus if len(d['content']) >= 100]
    jsonl(target / 'corpus.jsonl', corpus)
    eligible = [d for d in corpus if 600 <= len(d['content']) <= 6500]
    random.Random(20260908).shuffle(eligible)
    selected = eligible[:220]
    if len(selected) != 220:
        raise ValueError('Insufficient source topics')

    def generate(batch):
        i, docs = batch
        sources = []
        for d in docs:
            options = [s for s in re.split(r'(?<=[.!?])\s+', d['content']) if len(s) >= 40]
            sources.append({'id':d['id'], 'title':d['title'],
                            'evidence_options':[{'id':f'S{n:03}', 'text':s} for n,s in enumerate(options)]})
        result = api.json(f'rag_labeled_{i:03}', RAG_INSTRUCTION, sources, max_tokens=2500)['items']
        if [v['id'] for v in result] != [d['id'] for d in docs]:
            raise ValueError(f'Source ids/order mismatch in batch {i}')
        for item, doc, source in zip(result, docs, sources):
            evidence_id = item.pop('evidence_id')
            options = {o['id']:o['text'] for o in source['evidence_options']}
            if evidence_id not in options:
                raise ValueError(f'Invalid evidence id in {item["id"]}')
            item['evidence'] = options[evidence_id]
            assert item['evidence'] in doc['content']
            if not all(isinstance(item[k], str) and len(item[k]) > 4 for k in ['en', 'zh']):
                raise ValueError('Invalid generated question')
        return result

    batches = [(i, selected[n:n+5]) for i, n in enumerate(range(0, 220, 5))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        generated = [item for group in pool.map(generate, batches) for item in group]
    rag = []

    def add(family, index, kind, question, gold, evidence, **extra):
        # Split within each family type, before any embeddings or retrieval results.
        split = 'dev' if index % 5 == 0 else 'test'
        rag.append(dict(id=f'R{len(rag)+1:03}', family_id=family, split=split, kind=kind,
                        query=question, gold_groups=[[g] for g in gold], evidence=evidence, **extra))

    for i, row in enumerate(generated[:120]):
        for lang in ['en', 'zh']:
            add(f'single-{i:03}', i, 'answerable', row[lang], [row['id']], [row['evidence']], language=lang)
    for i in range(40):
        a, b = generated[120+2*i:122+2*i]
        for lang in ['en', 'zh']:
            query = ('I have two separate questions: ' if lang == 'en' else '我有两个不同的问题：') + a[lang] + ' ' + b[lang]
            add(f'multi-{i:03}', i, 'multi_evidence', query, [a['id'], b['id']], [a['evidence'], b['evidence']], language=lang)
    for i, row in enumerate(generated[200:]):
        add(f'context-{i:03}', i, 'needs_history', '刚才说的情况，具体应该怎么理解？', [row['id']], [row['evidence']], context=row['zh'], language='zh')
        add(f'context-{i:03}', i, 'context_supplied', row['zh'] + ' 具体应该怎么理解？', [row['id']], [row['evidence']], context=row['zh'], language='zh')
    # Personal records cannot be answered from a public health encyclopedia.
    private = ['上次血钾化验的精确数值', '昨天测得的血压', '本周血糖仪的平均读数', '预约门诊的具体时间',
               '我以前实际服用的药物剂量', '上个月的住院病案号', '我的医疗保险剩余额度', '我的手术日期',
               '昨天检验单上的肌酐数值', '我的疫苗接种日期', '我的药物过敏检测结果', '上周影像报告的原文',
               '我的主治医生姓名', '我下一次复诊的医院', '我的血型检验结果', '我的家族基因检测结果',
               '我最近一次体重测量值', '上次就诊支付的费用', '我的骨密度检查数值', '我的体检报告领取密码']
    unrelated = ['汽车发动机序列号', '笔记本电脑保修日期', '航班座位号', '银行卡开户网点', '宽带账号密码',
                 '图书馆借书截止日期', '上次购买冰箱的发票编号', '办公室打印机地址', '火车票检票口', '租赁合同到期日',
                 '房屋电表读数', '手机套餐剩余流量', '快递柜取件码', '网盘文件分享密码', '自行车车锁密码',
                 '相机镜头的购买价格', '游戏账号等级', '公司报销单审批人', '停车场车位编号', '上次网购订单编号']
    for i, phrase in enumerate(private + unrelated):
        add(f'negative-{i:03}', i, 'unanswerable', f'请告诉我{phrase}。', [], [], language='zh',
            reason='Personal information absent from the public corpus', negative_type='personal_medical' if i < 20 else 'out_of_domain')
    assert len(rag) == 400
    jsonl(target / 'rag_cases.jsonl', rag)

    def memory_batch(pair):
        i, category = pair
        prefix = f'M{i+1:02}-'
        cases = api.json(f'memory_{i:02}', MEMORY_INSTRUCTION, {'prefix': prefix, 'category': category})['scenarios']
        if [c['id'] for c in cases] != [f'{prefix}{n:02}' for n in range(1, 11)]:
            raise ValueError(f'Memory ids/order mismatch {category}')
        for n, c in enumerate(cases):
            if [set(t) for t in c['turns']] == [{'user'},{'assistant'},{'user'},{'assistant'}]:
                c['turns'] = [{**c['turns'][0], **c['turns'][1]}, {**c['turns'][2], **c['turns'][3]}]
            if len(c['turns']) != 2 or len(c['queries']) != 2:
                raise ValueError('Memory case requires 2 turns and 2 queries')
            if not all(q['expected_facts'] and q['required_groups'] and all(q['required_groups']) for q in c['queries']):
                raise ValueError('Missing expected memory facts')
            c.update(split='dev' if n % 5 == 0 else 'test', family_id=c['id'], synthetic=True)
        return cases

    with ThreadPoolExecutor(max_workers=4) as pool:
        mem = [c for group in pool.map(memory_batch, enumerate(MEMORY_CATEGORIES)) for c in group]
    assert len(mem) == 120
    jsonl(target / 'mem0_cases.jsonl', mem)
    # Gold review and schema checks occur before retrieval, preserving the sealed bytes afterwards.
    write(target / 'source_manifest.json', {
        'source_url': 'https://medlineplus.gov/xml/mplus_topics_compressed_2026-09-05.zip',
        'attribution': 'Source: MedlinePlus, National Library of Medicine',
        'rights': 'https://medlineplus.gov/about/using/usingcontent/',
        'content_scope': 'Health-topic summaries only; excludes licensed encyclopedias, drug monographs and images.',
        'source_snapshot': '2026-09-05', 'generation_model': 'qwen/qwen3.5-27b', 'seed': 20260908,
        'corpus_topics': len(corpus), 'rag_queries': 400, 'rag_families': 220,
        'rag_dev_queries': 80, 'rag_test_queries': 320, 'mem0_scenarios': 120,
        'mem0_dev_scenarios': 24, 'mem0_test_scenarios': 96,
        'label_scope': 'RAG target-source retrieval; exact evidence quotes validated. Memory fictional reported facts.',
        'clinical_review': False})
    print('Prepared 400 RAG probes and 120 memory scenarios; run review then freeze.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    main(parser.parse_args())
