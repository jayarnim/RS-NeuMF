import torch

def bpr(pos_score, neg_score):
    bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
    return bpr_loss