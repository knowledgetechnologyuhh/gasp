"""
code from: https://github.com/atsiami/STAViS/blob/master/models/sal_losses.py
code from: https://github.com/samyak0210/ViNet/blob/master/loss.py
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map

def logit(x):
    return np.log(x / (1 - x + 1e-08) + 1e-08)


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def cc_score(x, y, weights, batch_average=False, reduce=True):
    x = x.squeeze(1)
    x = torch.sigmoid(x)
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


def nss_score(x, y, weights, batch_average=False, reduce=True):
    x = x.squeeze(1)
    x = torch.sigmoid(x)
    y = y.squeeze(1)
    y = torch.gt(y, 0.0).float()

    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    std_x = torch.sqrt(torch.mean(torch.mean(torch.pow(torch.sub(x, mean_x), 2), 1, keepdim=True), 2, keepdim=True))
    x_norm = torch.div(torch.sub(x, mean_x), std_x)
    r_num = torch.sum(torch.sum(torch.mul(x_norm, y), 1, keepdim=True), 2, keepdim=True)
    r_den = torch.sum(torch.sum(y, 1, keepdim=True), 2, keepdim=True)
    r_val = torch.div(r_num, r_den + np.asscalar(np.finfo(np.float32).eps))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = -torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = -torch.sum(r_val)
        else:
            r_val = -r_val
    return r_val

def batch_image_sum(x):
    x = torch.sum(torch.sum(x, 1, keepdim=True), 2, keepdim=True)
    return x


def batch_image_mean(x):
    x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    return x


# def neg_ll_loss(x, y, weights, batch_average=False, reduce=True):
#     x = torch.softmax(x, dim=1)
#     y = y.squeeze(1)
#     y = torch.gt(y, 0.0).long()
#     final_loss = F.nll_loss(x,y, reduction="sum" if reduce else "none")
#     final_loss = final_loss * weights
#     if batch_average:
#         final_loss /= torch.sum(weights)
#     return final_loss

def neg_ll_loss(pred, target, weights, batch_average=False, reduce=True):
    batch_size = pred.size(0)

    output = pred.view(batch_size, -1)
    # output = torch.softmax(output, dim=1)
    label = target.view(batch_size, -1)
    label = torch.gt(label, 0.0).float()

    final_loss = F.nll_loss(output, torch.max(label, 1)[1], reduction="none").sum(0)
    final_loss = final_loss * weights

    if reduce:
        final_loss = torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss

def cross_entropy_loss(pred, target, weights, batch_average=False, reduce=True):
    batch_size = pred.size(0)
    output = pred.view(batch_size, -1)
    label = target.view(batch_size, -1)

    label = label / 255
    final_loss = F.binary_cross_entropy_with_logits(output, label, reduction="none").sum(1)
    final_loss = final_loss * weights

    if reduce:
        final_loss = torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss


def kl_div_loss(pred, target, weights, batch_average=False, reduce=True):
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

    pred = pred / (expand_s_map * 1.0)
    target = target / (expand_gt * 1.0)

    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)

    eps = 2.2204e-16
    final_loss = target * torch.log(eps + target / (pred + eps)) * weights
    # print(torch.log(eps + gt/(s_map + eps))   )
    if reduce:
        final_loss = torch.mean(torch.nansum(final_loss, 1))  # torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss


# METRIC
def aucj_metric_DEPRECATED(pred, gt, jitter=True, toPlot=False, normalize=False):
    # pred=saliencyMap is the saliency map
    # gt=fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve
    scores = []
    device = gt.device
    for saliencyMap, fixationMap in zip(pred, gt):
        # If there are no fixations to predict, return NaN
        if saliencyMap.size() != fixationMap.size():
            saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
            saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
            # saliencyMap = saliencyMap.cuda()
            # fixationMap = fixationMap.cuda()
        if len(saliencyMap.size())==3:
            saliencyMap = saliencyMap[0,:,:]
            fixationMap = fixationMap[0,:,:]
        if normalize:
            saliencyMap = normalize_map(saliencyMap)
        saliencyMap = saliencyMap.cpu().numpy()
        fixationMap = fixationMap.cpu().numpy()
        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            scores.append(score)
        # make the saliencyMap the size of the image of fixationMap

        if not np.shape(saliencyMap) == np.shape(fixationMap):
            from scipy.misc import imresize
            saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

        # jitter saliency maps that come from saliency models that have a lot of zero values.
        # If the saliency map is made with a Gaussian then it does not need to be jittered as
        # the values are varied and there is not a large patch of the same value. In fact
        # jittering breaks the ordering in the small values!
        if jitter:
            # jitter the saliency map slightly to distrupt ties of the same numbers
            saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

        # normalize saliency map
        saliencyMap = (saliencyMap - saliencyMap.min()) \
                      / (saliencyMap.max() - saliencyMap.min())

        if np.isnan(saliencyMap).all():
            print('NaN saliencyMap')
            score = float('nan')
            scores.append(score)

        S = saliencyMap.flatten()
        F = fixationMap.flatten()

        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)
        Npixels = len(S)

        allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
        tp = np.zeros((Nfixations + 2))
        fp = np.zeros((Nfixations + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(Nfixations):
            thresh = allthreshes[i]
            aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
            tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
            # above threshold
            fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
            # above threshold

        score = np.trapz(tp, x=fp)
        # allthreshes = np.insert(allthreshes, 0, 0)
        # allthreshes = np.append(allthreshes, 1)

        if toPlot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(saliencyMap, cmap='gray')
            ax.set_title('SaliencyMap with fixations to be predicted')
            [y, x] = np.nonzero(fixationMap)
            s = np.shape(saliencyMap)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')

            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()

        scores.append(score)
    return torch.tensor(np.nanmean(scores)).to(device)


def aucj_metric(pred, gt, jitter=True, toPlot=False, normalize=False):
    # pred=saliencyMap is the saliency map
    # gt=fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve
    scores = []
    device = gt.device
    for saliencyMap, fixationMap in zip(pred, gt):
        # If there are no fixations to predict, return NaN
        if saliencyMap.size() != fixationMap.size():
            saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
            saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
            # saliencyMap = saliencyMap.cuda()
            # fixationMap = fixationMap.cuda()
        if len(saliencyMap.size())==3:
            saliencyMap = saliencyMap[0,:,:]
            fixationMap = fixationMap[0,:,:]
        if normalize:
            saliencyMap = normalize_map(saliencyMap)
        saliencyMap = saliencyMap.cpu().numpy()
        fixationMap = fixationMap.cpu().numpy()
        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            scores.append(score)
        # make the saliencyMap the size of the image of fixationMap

        if not np.shape(saliencyMap) == np.shape(fixationMap):
            from scipy.misc import imresize
            saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

        # jitter saliency maps that come from saliency models that have a lot of zero values.
        # If the saliency map is made with a Gaussian then it does not need to be jittered as
        # the values are varied and there is not a large patch of the same value. In fact
        # jittering breaks the ordering in the small values!
        if jitter:
            # jitter the saliency map slightly to distrupt ties of the same numbers
            saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

        # normalize saliency map
        saliencyMap = (saliencyMap - saliencyMap.min()) \
                      / (saliencyMap.max() - saliencyMap.min())

        if np.isnan(saliencyMap).all():
            print('NaN saliencyMap')
            score = float('nan')
            scores.append(score)

        S = saliencyMap.flatten()
        F = fixationMap.flatten()

        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)
        Npixels = len(S)

        allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
        tp = np.zeros((Nfixations + 2))
        fp = np.zeros((Nfixations + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(Nfixations):
            thresh = allthreshes[i]
            aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
            tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
            # above threshold
            fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
            # above threshold

        score = np.trapz(tp, x=fp)
        # allthreshes = np.insert(allthreshes, 0, 0)
        # allthreshes = np.append(allthreshes, 1)

        if toPlot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(saliencyMap, cmap='gray')
            ax.set_title('SaliencyMap with fixations to be predicted')
            [y, x] = np.nonzero(fixationMap)
            s = np.shape(saliencyMap)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')

            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()

        scores.append(score)
    return torch.tensor(np.nanmean(scores)).to(device)
