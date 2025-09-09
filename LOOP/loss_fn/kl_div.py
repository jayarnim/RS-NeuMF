from torch.distributions import kl_divergence


def kl_div(approx, prior, mask=None):
    kl_tensor = kl_divergence(approx, prior)

    if mask is not None:
        mask = mask.to(kl_tensor.device)
        num_valid = mask.numel() - mask.sum()
        kl_tensor_masked = kl_tensor.masked_fill(mask, 0.0)
        kl_sum = kl_tensor_masked.sum()
        return kl_sum / (num_valid + 1e-8)

    else:
        return kl_tensor.mean()