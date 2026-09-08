"""Local BM25 and weighted reciprocal-rank fusion over a fixed document snapshot.

Vectors must be normalized. Embedding and optional query translation are supplied
by the caller so the index does not conceal external model calls.
"""
from collections import Counter, defaultdict
import re

import numpy as np


def tokens(text):
    # Keep abbreviations and numbers; CJK bigrams permit same-language lexical use.
    result = re.findall(r'[a-z0-9]+', text.casefold())
    for sequence in re.findall(r'[\u3400-\u9fff]+', text):
        result.extend(sequence[i:i+2] for i in range(max(1, len(sequence)-1)))
    return result


class HybridIndex:
    def __init__(self, documents, vectors, k1=1.2, b=0.75):
        self.documents = documents
        self.vectors = np.asarray(vectors)
        if self.vectors.ndim != 2 or len(documents) != len(self.vectors):
            raise ValueError('One vector required per document')
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-6):
            raise ValueError('Document vectors must be finite unit vectors')
        self.k1, self.b = k1, b
        terms = [Counter(tokens(d['content'])) for d in documents]
        self.lengths = np.asarray([sum(t.values()) for t in terms],dtype=float)
        self.average_length = self.lengths.mean()
        if self.average_length == 0:
            raise ValueError('Corpus has no indexable tokens')
        self.postings = defaultdict(list)
        for i, counts in enumerate(terms):
            for term, count in counts.items():
                self.postings[term].append((i,count))

    def lexical_scores(self, query):
        scores = np.zeros(len(self.documents))
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            indices, counts = np.asarray(postings).T
            idf = np.log1p((len(self.documents)-len(postings)+0.5)/(len(postings)+0.5))
            norm = self.k1*(1-self.b+self.b*self.lengths[indices]/self.average_length)
            scores[indices] += idf * counts*(self.k1+1)/(counts+norm)
        return scores

    @staticmethod
    def rank(values, positive_only=False):
        order = np.argsort(-values,kind='stable')
        return order[values[order] > 0] if positive_only else order

    @staticmethod
    def fuse(dense_order, sparse_order, size, dense_weight=0.75, rank_constant=60, window=50):
        if not 0 < dense_weight < 1 or rank_constant < 0 or window < 1:
            raise ValueError('Invalid RRF parameters')
        scores = np.zeros(size)
        for weight, order in [(dense_weight,dense_order),(1-dense_weight,sparse_order)]:
            selected = np.asarray(order[:window],dtype=int)
            scores[selected] += weight/(rank_constant+np.arange(1,len(selected)+1))
        return HybridIndex.rank(scores,positive_only=True), scores

    def search(self, query, query_vector, *, lexical_query=None, top_k=3,
               dense_weight=0.75, rank_constant=60, window=50):
        vector = np.asarray(query_vector)
        if vector.shape != self.vectors.shape[1:] or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-6):
            raise ValueError('Query vector must be a finite unit vector of matching dimension')
        dense = self.vectors @ vector
        sparse = self.lexical_scores(query if lexical_query is None else lexical_query)
        order, fused = self.fuse(self.rank(dense), self.rank(sparse,True),len(self.documents),
                                 dense_weight,rank_constant,window)
        return [dict(self.documents[int(i)], score=float(fused[i]),dense_score=float(dense[i]),
                     bm25_score=float(sparse[i])) for i in order[:top_k]]
