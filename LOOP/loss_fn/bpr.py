import torch.nn.functional as F

def bpr(pos, neg):
    return -F.logsigmoid(pos - neg).mean()