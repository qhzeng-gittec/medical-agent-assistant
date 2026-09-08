"""Summarize recorded provider usage, including calibration and format repairs."""
import numpy as np

from common import RESULTS, read, write


def main():
    phases = {}
    seen = set()
    translations = []
    for path in sorted(RESULTS.rglob('*.json')):
        value = read(path)
        if not isinstance(value,dict) or not {'request','response','status','seconds'}<=value.keys():
            continue
        response = value['response']
        identifier = response.get('id',path.relative_to(RESULTS).as_posix())
        if identifier in seen:
            continue
        seen.add(identifier)
        phase = path.relative_to(RESULTS).parts[0]
        group = phases.setdefault(phase,{'responses':0,'reported_cost_usd':0.,'responses_without_cost':0})
        group['responses'] += 1
        cost = response.get('usage',{}).get('cost')
        if isinstance(cost,(int,float)):
            group['reported_cost_usd'] += cost
        else:
            group['responses_without_cost'] += 1
        if phase=='retrieval' and path.stem.startswith('translate_'):
            translations.append(value['seconds'])
    result = {'phases':phases,'reported_cost_usd':sum(p['reported_cost_usd'] for p in phases.values()),
              'scope':'Recorded provider response usage only, including calibration/format repairs; excludes reused v2 vectors and Mem0 execution, and Codex subscription usage. Missing costs are unknown, not zero.',
              'translation_batch_seconds':{'n':len(translations),'median':float(np.median(translations)),
                                           'p95':float(np.quantile(translations,.95))} if translations else None,
              'latency_note':'Translation is batched and cached in this offline experiment. Batch wall times are not single-query production latency.'}
    write(RESULTS/'cost_summary.json',result)
    print(result)


if __name__=='__main__':
    main()
