import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def gspo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_log_ratio = (new_log_probs - old_log_probs).mean(dim=-1)
    sequence_ratio = sequence_log_ratio.exp()
    clipped_ratio = sequence_ratio.clamp(1 - clip_low, 1 + clip_high)
    objective = torch.minimum(sequence_ratio * advantages, clipped_ratio * advantages)
    return -objective.mean(), sequence_ratio, clipped_ratio


def main() -> None:
    rewards = torch.tensor([0.1, 0.4, 0.8, 0.7, 0.0])
    advantages = group_advantages(rewards)
    old_log_probs = torch.tensor(
        [
            [-1.1, -0.8, -1.2, -0.9],
            [-0.9, -1.0, -0.7, -1.1],
            [-1.2, -1.1, -0.8, -0.9],
            [-0.8, -0.9, -1.0, -1.2],
            [-1.0, -1.2, -0.9, -0.8],
        ]
    )
    policy_shift = torch.tensor([0.05, 0.15, 0.40, -0.35, -0.10]).unsqueeze(1)
    new_log_probs = old_log_probs + policy_shift
    loss, ratio, clipped_ratio = gspo_loss(old_log_probs, new_log_probs, advantages)

    print("rewards       =", rewards.tolist())
    print("advantages    =", [round(value, 4) for value in advantages.tolist()])
    print("sequence ratio=", [round(value, 4) for value in ratio.tolist()])
    print("clipped ratio =", [round(value, 4) for value in clipped_ratio.tolist()])
    print("policy loss   =", round(loss.item(), 6))

    if not torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6):
        raise RuntimeError("Group-relative advantages should have zero mean.")
    if not torch.all((clipped_ratio >= 0.8) & (clipped_ratio <= 1.3)):
        raise RuntimeError("Clipped sequence ratio is outside the configured interval.")


if __name__ == "__main__":
    main()
