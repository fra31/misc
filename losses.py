import torch


def cospgd_loss(pred, target, reduction='mean'):
    """Implementation of the loss for semantic segmentation from
    https://arxiv.org/abs/2302.02213.

    pred: B x cls x h x w
    target: B x h x w
    """

    sigm_pred = torch.sigmoid(pred)
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    w = (sigm_pred * y).sum(1) / sigm_pred.norm(p=2, dim=1)
    loss = F.cross_entropy(pred, target, reduction='none')
    loss = w * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss
