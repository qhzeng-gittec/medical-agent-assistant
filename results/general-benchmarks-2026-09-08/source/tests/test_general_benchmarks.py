import torch

from prepare_general_benchmarks import convert
from run_general_benchmarks import parse_letter, target_logprobs


def test_arc_numeric_labels_preserve_source_order():
    row = convert('arc_challenge', dict(id='example', question='Which one?',
        choices={'label': ['1', '2', '3'], 'text': ['red', 'green', 'blue']}, answerKey='2'), 0)
    assert row['gold'] == 1
    assert row['candidates'] == ['red', 'green', 'blue']


def test_hellaswag_cleanup_and_reference():
    row = convert('hellaswag', dict(ind=2, activity_label='Run', ctx_a='A person [title]',
        ctx_b='runs', endings=['stops [noise]', 'walks'], label='1'), 0)
    assert row['question'] == 'Run: A person. Runs'
    assert row['candidates'] == ['stops ', 'walks']
    assert row['gold'] == 1


def test_causal_likelihood_excludes_context_and_last_logit():
    logits = torch.tensor([[100., 0., 0.], [0., 2., 0.], [0., 0., 3.], [100., 0., 0.]])
    actual = target_logprobs(logits, [1, 2])
    expected = torch.tensor([2. - torch.logsumexp(logits[1], 0), 3. - torch.logsumexp(logits[2], 0)])
    assert torch.allclose(actual, expected)
    assert torch.allclose(target_logprobs(logits[-3:], [1, 2]), expected)


def test_letter_parser_does_not_match_prose_or_unavailable_option():
    assert parse_letter('A reasonable explanation', 4) is None
    assert parse_letter('Answer: **B**.', 4) == 'B'
    assert parse_letter('Because of friction', 4) is None
    assert parse_letter('E', 4) is None
    assert parse_letter('C.', 3) == 'C'
