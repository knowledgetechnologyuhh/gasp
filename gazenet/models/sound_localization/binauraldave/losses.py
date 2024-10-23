"""
code from: https://github.com/samyak0210/ViNet/blob/master/loss.py
"""

import torch
import numpy as np


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def kl_div_loss(pred, target, weights):
    assert pred.size() == target.size()
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    batch_size = pred.size(0)
    w = pred.size(1)
    h = pred.size(2)

    sum_s_map = torch.sum(pred.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == pred.size()

    sum_gt = torch.sum(target.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == target.size()

    pred = pred / (expand_s_map * weights)
    target = target / (expand_gt * weights)

    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)

    eps = 2.2204e-16
    result = target * torch.log(eps + target / (pred + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))


def cc_score(x, y, weights, batch_average=False, reduce=True):
    x = x.squeeze(1)
    #x = torch.sigmoid(x)
    y = y.squeeze(1)
    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    mean_y = torch.mean(torch.mean(y, 1, keepdim=True), 2, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(torch.sum(torch.mul(xm, ym), 1, keepdim=True), 2, keepdim=True)
    r_den_x = torch.sum(torch.sum(torch.mul(xm, xm), 1, keepdim=True), 2, keepdim=True)
    r_den_y = torch.sum(torch.sum(torch.mul(ym, ym), 1, keepdim=True), 2, keepdim=True) + np.asscalar(
        np.finfo(np.float32).eps)
    r_val = torch.div(r_num, torch.sqrt(torch.mul(r_den_x, r_den_y)))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val


def sim_score(x, y, weights, batch_average=False, reduce=True):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    batch_size = x.size(0)
    w = x.size(2)
    h = x.size(3)

    x = normalize_map(x.squeeze(1))
    y = normalize_map(y.squeeze(1))

    sum_s_map = torch.sum(x.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == x.size()

    sum_gt = torch.sum(y.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    x = x / (expand_s_map * 1.0)
    y = y / (expand_gt * 1.0)

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    r_val = torch.sum(torch.min(x, y), 1)
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val
