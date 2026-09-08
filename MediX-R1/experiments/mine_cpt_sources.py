"""Index downloaded medical prose and retrieve evidence for unresolved text questions."""
import argparse
import hashlib
import json
import re
import sqlite3
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from evaluation.audit_cpt_reuse import anchors, normalized, query_parts
from common.io import load_jsonl, save_jsonl
from data_processing.download_cpt_sources import ROOT, save

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/cpt_expanded_coverage_v1'
STOP = set(ENGLISH_STOP_WORDS) | set('medical question patient likely diagnosis following year old man woman history develops shows answer best cause most months days years'.split())


def connect(dataset):
    connection = sqlite3.connect(ROOT / f'{dataset}.sqlite3')
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA cache_size=-65536')
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(id UNINDEXED, path UNINDEXED, title, body, tokenize='porter unicode61')")
    connection.execute('CREATE TABLE IF NOT EXISTS inputs(path TEXT PRIMARY KEY, sha256 TEXT, rows INTEGER)')
    return connection


def xml_passages(text, member):
    root = ET.fromstring(text)
    title = root.findtext('.//book-part-meta/title-group/title') or root.findtext('.//title') or member
    for sec in root.findall('.//body//sec'):
        heading = ''.join(sec.find('title').itertext()) if sec.find('title') is not None else ''
        if re.search(r'references|review questions|disclosure|continuing education', heading, re.I):
            continue
        for index, child in enumerate(sec):
            if child.tag not in ['p', 'list', 'table-wrap']:
                continue
            body = re.sub(r'\s+', ' ', ' '.join(child.itertext())).strip()
            if len(body) < 80:
                continue
            yield (f"{member}:{sec.get('id', heading)}:{index}", member, title+' -- '+heading, body)


def index_dataset(dataset, watch):
    connection = connect(dataset)
    started = time.time()
    while True:
        status = json.loads((ROOT/f'{dataset}_status.json').read_text(encoding='utf-8'))
        paths = ([ROOT/'statpearls/statpearls_NBK430685.tar.gz'] if dataset == 'statpearls'
                 else sorted((ROOT/dataset/'chunk').glob('*.jsonl')))
        for path in paths:
            receipt = path.with_suffix(path.suffix+'.receipt.json')
            if not receipt.exists():
                continue
            receipt_data = json.loads(receipt.read_text(encoding='utf-8'))
            relative = str(path.relative_to(ROOT))
            existing = connection.execute('SELECT sha256 FROM inputs WHERE path=?', (relative,)).fetchone()
            if existing:
                if existing[0] != receipt_data['actual_sha256']:
                    raise ValueError(f'Indexed source changed: {relative}')
                continue
            count = 0
            with connection:
                if dataset == 'statpearls':
                    with tarfile.open(path, 'r|gz') as archive:
                        for member in archive:
                            if member.isfile() and member.name.endswith('.nxml'):
                                rows = list(xml_passages(archive.extractfile(member).read(), member.name))
                                connection.executemany('INSERT INTO passages VALUES(?,?,?,?)', rows)
                                count += len(rows)
                else:
                    with path.open(encoding='utf-8') as file:
                        batch = []
                        for line in file:
                            row = json.loads(line)
                            batch.append((row['id'], relative, row['title'], row['content']))
                            count += 1
                            if len(batch) == 1000:
                                connection.executemany('INSERT INTO passages VALUES(?,?,?,?)', batch)
                                batch = []
                        connection.executemany('INSERT INTO passages VALUES(?,?,?,?)', batch)
                connection.execute('INSERT INTO inputs VALUES(?,?,?)', (relative, receipt_data['actual_sha256'], count))
            files, total = connection.execute('SELECT count(*), coalesce(sum(rows),0) FROM inputs').fetchone()
            progress = dict(dataset=dataset, indexed_files=files, indexed_passages=total, elapsed_seconds=round(time.time()-started), download_state=status['state'])
            save(ROOT/f'{dataset}_index_status.json', progress)
            print(json.dumps(progress), flush=True)
        if status['state'] == 'failed':
            raise RuntimeError(f'{dataset} download failed; index is incomplete')
        if not watch or status['state'] == 'complete':
            break
        time.sleep(15)
    connection.close()
    if status['state'] == 'complete':
        progress = json.loads((ROOT/f'{dataset}_index_status.json').read_text(encoding='utf-8'))
        progress['state'] = 'complete'
        progress['download_state'] = 'complete'
        save(ROOT/f'{dataset}_index_status.json', progress)


def queries(row, frequency):
    parts = query_parts(row)
    terms = {t for t in normalized(' '.join(parts)).split() if not t.isdigit() and len(t)>2} - STOP
    ordered = sorted(terms, key=lambda t: (frequency[t], -len(t), t))
    answer = sorted(anchors(row)-STOP, key=lambda t: (frequency[t], -len(t), t))
    quote = lambda word: '"'+word+'"'
    result = [' OR '.join(map(quote, ordered[:14]))]
    if answer:
        result.append(' OR '.join(map(quote, answer[:6])))
    if len(ordered) >= 2:
        result.append(' AND '.join(map(quote, ordered[:2])))
    return [q for q in result if q]


def retrieve(datasets):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_jsonl(LAB/'outputs/cpt_reuse_inventory_v1/review_queue.jsonl') if r['task'] in ['knowledge', 'case']]
    assert len(rows) == 1353
    for row in rows:
        row['user_text'] = row['question']
    frequency = Counter(t for r in rows for t in set(normalized(' '.join(query_parts(r))).split()))
    connections = {name: sqlite3.connect(f'file:{(ROOT/(name+".sqlite3")).as_posix()}?mode=ro', uri=True) for name in datasets}
    snapshots = {}
    for name, connection in connections.items():
        connection.execute('BEGIN')
        inputs = connection.execute('SELECT path,sha256,rows FROM inputs ORDER BY path').fetchall()
        snapshots[name] = dict(indexed_files=len(inputs), indexed_passages=sum(r[2] for r in inputs),
                               inputs_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest())
    # Textbooks and StatPearls are small enough for an exhaustive sparse search.
    # PubMed stays in the disk index because its full corpus is tens of millions of rows.
    compact = []
    for name, connection in connections.items():
        if name != 'pubmed':
            compact.extend(dict(dataset=name, passage_id=sid, path=path, title=title, text=body)
                           for sid,path,title,body in connection.execute('SELECT id,path,title,body FROM passages'))
    global_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1,
                                        max_features=180000, sublinear_tf=True, dtype=np.float32)
    compact_matrix = global_vectorizer.fit_transform(c['title']+'\n'+c['text'] for c in compact) if compact else None
    print(f'Exhaustive compact-corpus index: {len(compact)} passages', flush=True)
    results = []
    for index, row in enumerate(rows):
        candidates = {}
        if compact:
            query = global_vectorizer.transform(query_parts(row))
            scores = (compact_matrix @ (.35*query[0]+.45*query[1]+.2*query[2]).T).toarray().ravel()
            for i in np.argpartition(-scores, min(30, len(scores)-1))[:30]:
                candidate = compact[int(i)]
                candidates[hashlib.sha256(normalized(candidate['text']).encode()).hexdigest()] = dict(candidate, global_score=float(scores[i]))
        for name, connection in connections.items():
            for query in queries(row, frequency):
                for sid, path, title, body in connection.execute('SELECT id,path,title,body FROM passages WHERE passages MATCH ? ORDER BY bm25(passages,0,0,1,1) LIMIT 16', (query,)):
                    fingerprint = hashlib.sha256(normalized(body).encode()).hexdigest()
                    candidates.setdefault(fingerprint, dict(dataset=name, passage_id=sid, path=path, title=title, text=body))
        choices = list(candidates.values())
        if choices:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True, dtype=np.float32)
            matrix = vectorizer.fit_transform([c['title']+'\n'+c['text'] for c in choices]+query_parts(row))
            scores = (matrix[:-3] @ (.35*matrix[-3]+.45*matrix[-2]+.2*matrix[-1]).T).toarray().ravel()
            # Bare headings and figure labels otherwise outrank explanatory paragraphs.
            scores *= np.array([min(1., len(c['text'].split())/60) for c in choices])
            best = np.argsort(-scores)[:5]
            ranked = []
            for i in best:
                candidate = choices[int(i)]
                terms = anchors(row)
                hits = terms.intersection(normalized(candidate['text']).split())
                ranked.append(dict(candidate, local_rerank_score=float(scores[i]), answer_anchor_hits=sorted(hits),
                                   answer_anchor_fraction=len(hits)/len(terms) if terms else None))
        else:
            ranked = []
        results.append(dict(source_id=row['source_id'], task=row['task'], split=row['split'], question=row['question'],
                            target=row['target'], reasoning_content=row['reasoning_content'], candidates=ranked,
                            semantic_coverage='pending_review', training_ready=False))
        if index % 100 == 0:
            print(f'Retrieved {index+1}/{len(rows)}', flush=True)
    for connection in connections.values():
        connection.close()
    review_path = OUT/'reviewed_new_evidence.jsonl'
    reviewed = {r['source_id']:r for r in load_jsonl(review_path)} if review_path.exists() else {}
    for row in results:
        if row['source_id'] in reviewed:
            evidence = reviewed[row['source_id']]
            if normalized(evidence['target']) != normalized(row['target']):
                raise ValueError('Reviewed evidence target does not match the current question.')
            row['reviewed_evidence'] = evidence
            row['semantic_coverage'] = 'reviewed_existing_prose_pending_token_inclusion'
    save_jsonl(OUT/'priority_evidence.jsonl', results)
    summary = dict(required_priority_questions=1353, by_task=dict(Counter(r['task'] for r in rows)),
                   datasets=datasets, source_snapshots=snapshots, questions_with_candidates=sum(bool(r['candidates']) for r in results),
                   semantic_reviewed=sum(r['source_id'] in reviewed for r in results), model_api_calls=0, ready_for_training=False,
                   note='BM25 and local TF-IDF rank candidates only. Exact original excerpts, no generated prose. Full-coverage training requires semantic and token inclusion audits.')
    save(OUT/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['index', 'retrieve'])
    parser.add_argument('--datasets', nargs='+', choices=['textbooks','statpearls','pubmed'], required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.action == 'index':
        for dataset in args.datasets:
            index_dataset(dataset, args.watch)
    else:
        retrieve(args.datasets)
