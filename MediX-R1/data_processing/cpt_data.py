"""Bound document contributions, then pack an EOS-delimited causal token stream."""
import hashlib
import random


def sample_spans(input_ids: list[int], document_id: str, max_tokens: int,
                 window_length: int, seed: int) -> list[tuple[int, int]]:
    """Return source offsets; the budget includes one EOS per sampled excerpt."""
    if max_tokens < 2 or window_length < 2:
        raise ValueError('Token budgets must allow text and EOS.')
    if len(input_ids) + 1 <= max_tokens:
        return [(0, len(input_ids))]
    count = (max_tokens + window_length - 1) // window_length
    # Each disjoint stratum contributes a contiguous window, not always its prefix.
    rng = random.Random(f'{seed}:{document_id}')
    spans = []
    for index in range(count):
        budget = max_tokens // count + (index < max_tokens % count)
        left = len(input_ids) * index // count
        right = len(input_ids) * (index + 1) // count
        width = budget - 1
        start = rng.randint(left, right - width)
        spans.append((start, start + width))
    return spans


def pack_excerpts(excerpts: list[dict], max_length: int) -> list[dict]:
    """Concatenate without overlap or dropped remainders, retaining source offsets.

    Attention and recurrent state continue across EOS inside a pack. This is
    ordinary causal pretraining, not attention-isolated SFT sequence packing.
    """
    if max_length < 3:
        raise ValueError('max_length must be at least 3 to retain any remainder without singleton rows.')
    remaining = sum(len(row['input_ids']) for row in excerpts)
    if remaining < 2:
        raise ValueError('A training stream needs at least two tokens.')
    blocks = []
    block = dict(input_ids=[], spans=[])
    target = max_length - 1 if remaining == max_length + 1 else min(max_length, remaining)
    for excerpt in excerpts:
        ids = excerpt['input_ids']
        offset = 0
        while offset < len(ids):
            take = min(target - len(block['input_ids']), len(ids) - offset)
            start = len(block['input_ids'])
            block['input_ids'].extend(ids[offset:offset + take])
            content_length = len(ids) - 1  # Each excerpt ends in exactly one appended EOS.
            block['spans'].append(dict(
                document_id=excerpt['document_id'], source=excerpt['source'], bucket=excerpt['bucket'],
                block_start=start, block_end=start + take,
                source_token_start=excerpt['source_token_start'] + min(offset, content_length),
                source_token_end=excerpt['source_token_start'] + min(offset + take, content_length),
                ends_with_eos=offset + take == len(ids),
                is_document_end=offset + take == len(ids) and excerpt['is_document_end']))
            offset += take
            remaining -= take
            if len(block['input_ids']) == target:
                blocks.append(block)
                block = dict(input_ids=[], spans=[])
                # Avoid a one-token final row, which has no causal prediction target.
                target = max_length - 1 if remaining == max_length + 1 else min(max_length, remaining)
    return blocks


def token_fingerprint(input_ids: list[int]) -> str:
    return hashlib.sha256(b''.join(token.to_bytes(4, 'little') for token in input_ids)).hexdigest()
