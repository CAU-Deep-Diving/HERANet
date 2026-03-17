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
    원래 augmentation.txt 로직을 그대로 유지하되
    notebook에서 바로 실행 가능하도록 포함한 버전
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

    # token dropout
    dropout_mask = (torch.rand_like(inputs.float()) > p_drop).long()
    inputs = inputs * dropout_mask

    # token mask
    mask = (torch.rand_like(inputs.float()) < p_mask) & (inputs != pad_idx)
    inputs[mask] = unk_idx

    # local shuffle / random swap
    for seq in inputs:
        seq_len = (seq != pad_idx).sum().item()

        for i in range(seq_len - 1):
            if torch.rand(1).item() < p_shuffle:
                tmp = seq[i].item()
                seq[i] = seq[i + 1]
                seq[i + 1] = tmp

        if seq_len > 2 and torch.rand(1).item() < p_swap:
            idx = torch.randperm(seq_len)[:2]
            i, j = idx[0].item(), idx[1].item()
            tmp = seq[i].item()
            seq[i] = seq[j]
            seq[j] = tmp

    return inputs, lengths, labels