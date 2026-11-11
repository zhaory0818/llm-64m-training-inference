"""Optional right-padding removal, preserving original SFT input/label prefixes."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DynamicRightPadding:
    """Stack original fixed-width items, then remove a common unused PAD suffix.

    Standard items are (input_ids, labels). An optional third field gives the
    authoritative pre-padding length. Without it, PAD=0 is the right-padding
    sentinel; any supervised PAD is retained. No internal PAD or ignored real
    context is removed. Original truncation and label shifting are untouched.
    """
    pad_id: int = 0
    ignore_index: int = -100
    multiple: int = 8

    def __post_init__(self):
        if self.multiple < 1:
            raise ValueError('multiple must be positive')

    def __call__(self, batch):
        if not batch:
            raise ValueError('empty batch')
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError('matching [batch, time] inputs and labels required')
        if inputs.shape[1] < 2:
            raise ValueError('at least two positions required for next-token loss')
        occupied = (inputs != self.pad_id) | (labels != self.ignore_index)
        positions = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
        required = int((occupied * positions).max().item())
        if all(len(item) == 3 for item in batch):
            lengths = [int(item[2]) for item in batch]
            if any(n < 0 or n > inputs.shape[1] for n in lengths):
                raise ValueError('invalid authoritative length')
            required = max(required, max(lengths))
        elif any(len(item) != 2 for item in batch):
            raise ValueError('length metadata must be present for every item or none')
        width = min(inputs.shape[1], max(2, ((required + self.multiple - 1) // self.multiple) * self.multiple))
        if not torch.all(inputs[:, width:] == self.pad_id) or not torch.all(labels[:, width:] == self.ignore_index):
            raise AssertionError('attempt to remove real or supervised positions')
        return inputs[:, :width].contiguous(), labels[:, :width].contiguous()
