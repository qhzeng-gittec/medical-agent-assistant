import math
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from types import SimpleNamespace

import torch

from training.gspo import (collate_responses, format_score, microbatches, policy_loss, response_logprobs,
                                  score_rollouts, select_training_rows, trim_completion)


def test_sequence_ratio_is_geometric_mean_and_length_normalized():
    old = torch.tensor([-2.0, -2.0])
    new = old + torch.tensor([math.log(1.2), math.log(0.8)])
    _, ratio = policy_loss(new, old, 1.0)
    _, repeated = policy_loss(new.repeat(3), old.repeat(3), 1.0)
    assert torch.isclose(ratio, torch.tensor(math.sqrt(0.96)))
    assert torch.isclose(ratio, repeated)


def test_zero_mean_loss_still_has_policy_gradient():
    old = torch.tensor([-2.0, -3.0])
    good = old.clone().requires_grad_()
    bad = old.clone().requires_grad_()
    loss = (policy_loss(good, old, 1.0)[0] + policy_loss(bad, old, -1.0)[0]) / 2
    assert loss.item() == 0
    loss.backward()
    assert torch.all(good.grad < 0)
    assert torch.all(bad.grad > 0)


def test_clipping_only_removes_incentive_in_rewarded_direction():
    old = torch.tensor([-2.0])
    for advantage, shift, clipped in [(1, 0.1, True), (-1, -0.1, True),
                                      (1, -0.1, False), (-1, 0.1, False)]:
        new = (old + shift).requires_grad_()
        loss, _ = policy_loss(new, old, advantage)
        loss.backward()
        assert (new.grad.item() == 0) == clipped


def test_native_format_requires_complete_untruncated_response():
    assert format_score('Visible finding. </think> Final answer.', True) == 1
    assert format_score('Visible finding. </think> Final answer.', False) == 0
    assert format_score('Visible finding. </think>', True) == 0
    assert format_score('Visible finding. </think> Answer. </think>', True) == 0


def test_judge_batching_preserves_question_groups():
    scores = [0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 1, 2, 1, 0, 1] * 2
    batch = [dict(row={"source_id": str(i)}, public={"format_score": 1}) for i in range(32)]
    concurrent_calls = Barrier(2)

    def judge(rows, responses, output_dir, model):
        concurrent_calls.wait(timeout=5)
        return [{"id": row["source_id"], "score": scores[int(row["source_id"])]} for row in rows]

    with patch("training.gspo.judge_responses", side_effect=judge) as mocked:
        score_rollouts(batch, 4, 16, Path("unused"), "test")
    assert [len(call.args[0]) for call in mocked.call_args_list] == [16, 16]
    assert all(item["advantage"] == 0 for item in batch[:8])
    for offset in range(0, 32, 4):
        assert abs(sum(item["advantage"] for item in batch[offset:offset + 4])) < 1e-5
    assert batch[8]["advantage"] < 0 < batch[10]["advantage"]
    assert [item["public"]["judge"]["id"] for item in batch] == [str(i) for i in range(32)]


def test_failed_judge_aborts_before_assigning_rewards():
    batch = [dict(row={"source_id": str(i)}, public={"format_score": 1}) for i in range(8)]

    def judge(rows, responses, output_dir, model):
        if rows[0]["source_id"] == "4":
            raise RuntimeError("judge unavailable")
        return [{"id": row["source_id"], "score": 2} for row in rows]

    with patch("training.gspo.judge_responses", side_effect=judge):
        try:
            score_rollouts(batch, 4, 4, Path("unused"), "test")
        except RuntimeError as error:
            assert str(error) == "judge unavailable"
        else:
            raise AssertionError("Judge failure must propagate")
    assert all("reward" not in item["public"] and "advantage" not in item for item in batch)


def test_training_selection_covers_images_before_reusing_them():
    rows = [dict(source_id=str(i), source_group=group) for i, group in enumerate(["A", "A", "B", "C", "B"])]
    assert [r["source_id"] for r in select_training_rows(rows, 5)] == ["0", "2", "3", "1", "4"]
    assert len({r["source_id"] for r in select_training_rows(rows, 4)}) == 4


def test_batched_logprobs_exclude_prompt_and_left_padding():
    def item(prompt, reply):
        return dict(inputs=dict(input_ids=torch.tensor([prompt]), mm_token_type_ids=torch.zeros(1, len(prompt), dtype=torch.long),
                                pixel_values=torch.zeros(1, 3), image_grid_thw=torch.ones(1, 3, dtype=torch.long)),
                    completion_ids=reply)

    class Model:
        device = "cpu"

        def __call__(self, input_ids, logits_to_keep, **kwargs):
            logits = torch.nn.functional.one_hot((input_ids + 1) % 5, 5).float() * 2
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    batch = [item([1, 2], [3, 4]), item([0], [2])]
    inputs = collate_responses(batch, 0)
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    result = response_logprobs(Model(), batch, 0)
    denominator = math.log(math.exp(2) + 4)
    assert torch.allclose(result[0], torch.tensor([2 - denominator, 2 - denominator]))
    assert torch.allclose(result[1], torch.tensor([-denominator]))


def test_generation_padding_is_removed_after_first_eos():
    assert trim_completion([3, 4, 9, 0, 0], 9) == [3, 4, 9]
    assert trim_completion([3, 4, 5], 9) == [3, 4, 5]


def test_long_completions_reduce_actual_batch_without_changing_order():
    rows = [dict(completion_ids=list(range(length)), id=i) for i, length in enumerate([40, 40, 512, 512, 40, 40, 40, 40])]
    chunks = list(microbatches(rows, 4))
    assert [len(chunk) for chunk in chunks] == [2, 2, 4]
    assert [row["id"] for chunk in chunks for row in chunk] == list(range(8))
