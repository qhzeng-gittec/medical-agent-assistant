from collections import Counter

from transformers import StoppingCriteria


def repeated_line(text: str) -> bool:
    lines = [' '.join(line.split()) for line in text.splitlines()]
    return any(count >= 6 for line, count in Counter(lines).items() if len(line) >= 50)


class RepetitionStop(StoppingCriteria):
    """Single-sample calibration: check recent output every 64 generated tokens."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[1] - self.prompt_length
        if generated >= 512 and generated % 64 == 0:
            text = self.tokenizer.decode(input_ids[0, max(self.prompt_length, input_ids.shape[1] - 2048):].tolist(), skip_special_tokens=False)
            self.triggered = repeated_line(text)
        return self.triggered
