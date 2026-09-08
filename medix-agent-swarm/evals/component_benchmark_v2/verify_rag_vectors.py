"""Recompute all published rankings from real API vectors, without an API key."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(root):
    order=read(root/'vector_order.json')
    with np.load(root/'vectors.npz',allow_pickle=False) as vectors:
        docs=vectors['corpus']
        assert docs.shape==(len(order['corpus']),4096)
        for split in ['dev','test']:
            queries=vectors[split]
            assert queries.shape==(len(order[split]),4096)
            similarities=queries @ docs.T
            for case_id,values in zip(order[split],similarities):
                expected=read(root/'cases'/f'{case_id}.json')['ranking']
                indices=np.argsort(-values,kind='stable')[:10]
                assert [order['corpus'][i] for i in indices]==[h['id'] for h in expected],case_id
                np.testing.assert_allclose(values[indices],[h['score'] for h in expected],rtol=0,atol=1e-12)
    print('Verified all 400 Top-10 rankings and cosine scores from published API vectors.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args().results)
