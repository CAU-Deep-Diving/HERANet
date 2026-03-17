import torch


def apply_augmentation(
    inputs,
    lengths,
    labels,
    pad_idx=0,
    unk_idx=1,
    base_p_drop=0.05,
    base_p_mask=0.05,
    base_p_shuffle=0.05,
    base_p_swap=0.05,
    current_epoch: int = 0,
    total_epochs: int = 30,
):
    """
    epoch 진행률에 따라 augmentation 강도 조절
    """
    inputs = inputs.clone()

    progress = current_epoch / max(total_epochs, 1)

    if progress < 0.3:
        p_drop = p_mask = p_shuffle = p_swap = 0.0
    elif progress < 0.7:
        ratio = (progress - 0.3) / 0.4
        p_drop = base_p_drop * ratio
        p_mask = base_p_mask * ratio
        p_shuffle = base_p_shuffle * ratio
        p_swap = base_p_swap * ratio
    else:
        p_drop = base_p_drop
        p_mask = base_p_mask
        p_shuffle = base_p_shuffle
        p_swap = base_p_swap

    dropout_mask = (torch.rand_like(inputs.float()) > p_drop).long()
    inputs = inputs * dropout_mask

    mask = (torch.rand_like(inputs.float()) < p_mask) & (inputs != pad_idx)
    inputs[mask] = unk_idx

    for seq in inputs:
        seq_len = (seq != pad_idx).sum().item()

        for i in range(seq_len - 1):
            if torch.rand(1).item() < p_shuffle:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]

        if seq_len > 2 and torch.rand(1).item() < p_swap:
            i, j = torch.randint(0, seq_len, (2,))
            seq[i], seq[j] = seq[j], seq[i]

    return inputs, lengths, labels