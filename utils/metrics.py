# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

def box_dious(blist1, blist2):
    # Get the minimum bounding box (including region proprosal and ground truth) #
    # n = torch.rand([3, 4])
    # m = torch.rand([5, 4])
    n = blist1.cuda()
    m = blist2.cuda()

    #print("n.shape[0],n.shape[1]:", n.shape[0], n.shape[1])
    #print("m.shape[0],m.shape[1]:", m.shape[0], m.shape[1])
    nd0 = n.shape[0]
    md0 = m.shape[0]
    #print("##################################################################-------->>")
    nss = n.unsqueeze(1)
    nss = nss.expand(nd0, md0, 4)
    #print("n & n.shape:\n", n, "\n", n.shape)
    mss = m.unsqueeze(0)
    mss = mss.expand(nd0, md0, 4)
    #print("m & m.shape:\n", m, "\n", m.shape)
    nms = torch.cat((nss, mss), dim=2)
    #print("nms & nms.shape:\n", nms, "\n", nms.shape)

    A = nms[:, :, [0, 4]]
    B = nms[:, :, [1, 5]]
    C = nms[:, :, [2, 6]]
    D = nms[:, :, [3, 7]]
    Am = torch.min(A, 2)[0]
    Bm = torch.min(B, 2)[0]
    Cm = torch.max(C, 2)[0]
    Dm = torch.max(D, 2)[0]
    # AB = torch.cat((Am, Bm), dim=1).cuda()
    # CD = torch.cat((Cm, Dm), dim=1).cuda()
    XY = torch.zeros([Am.shape[0], Am.shape[1], 4]).cuda()
    XY[:, :, 0] = Am
    XY[:, :, 1] = Bm
    XY[:, :, 2] = Cm
    XY[:, :, 3] = Dm
    XYx = (XY[:, :, [2, 3]] - XY[:, :, [0, 1]]) ** 2
    XxY = XYx[:, :, 0] + XYx[:, :, 1]
    XYs = XxY.sqrt()  ###########################-> to get square root


    #######################################################
    #######################################################
    # The average distance between GT and RP is obtained #
    nms = torch.cat((n, m), dim=0)
    #print("nms & nms.shape:\n", nms, "\n", nms.shape)
    #########################################################
    #print("n.shape & n:\n", n.shape, "\n", n)
    n0 = n[:, [0, 3]]  # .unsqueeze(1)
    n1 = n[:, [1, 2]]  # .unsqueeze(1)
    #print("n0 & n0.shape:\n", n0, "\n", n0.shape)
    #print("n1 & n1.shape:\n", n1, "\n", n1.shape)
    ns = torch.cat((n, n0, n1), dim=1)
    #print("ns.shape & ns:\n", ns.shape, "\n", ns)
    ######################################################
    #print("m.shape & m:\n", m.shape, "\n", m)
    m0 = m[:, [0, 3]]  # .unsqueeze(1)
    m1 = m[:, [1, 2]]  # .unsqueeze(1)
    #print("m0 & m0.shape:\n", m0, "\n", m0.shape)
    #print("m1 & m1.shape:\n", m1, "\n", m1.shape)
    ms = torch.cat((m, m0, m1), dim=1)
    #print("ms.shape & ms:\n", ms.shape, "\n", ms)
    ################################################################
    ns = ns.unsqueeze(1)
    ms = ms.unsqueeze(0)
    #print("ns.shape & ns->unsqueeze:\n", ns.shape, "\n", ns)
    #print("ms.shape & ms->unsqueeze:\n", ms.shape, "\n", ms)

    n = ns
    m = ms
    #print("n.shape & n:\n", n.shape, "\n", n)
    #print("m.shape & m:\n", m.shape, "\n", m)
    tmp = (n - m) ** 2
    #print("tmp.shape:\n", tmp.shape)
    #print("tmp1->(n - m) ** 2:\n", tmp)
    # tmp = tmp[0:-1:2] + tmp[1:-1:2]
    # print(tmp[:,:,0::2],"\n", tmp[:,:,1::2])
    tmps = tmp[:, :, 0::2] + tmp[:, :, 1::2]
    #print("tmps and tmps.shape:\n", tmps, "\n", tmps.shape)
    # tmp = np.sqrt(tmps)               #######################-> to get square root
    tmp = tmps.sqrt()
    #print("tmp3->tmps-square root:\n", tmp, "\n", tmp.shape)
    # tmp = tmp.mean(axis=2, keepdim=False)/4
    tmp = torch.mean(tmp, dim=2, keepdim=False) / 4
    #print("tmp->mean:\n", tmp, "\n", tmp.shape)

    # get DIoU+ #
    #print("DIoU+ and DIoU.shape:\n", tmp / XYs, "\n", (tmp / XYs).shape)
    tmpx = (tmp / XYs).cuda()
    return (tmpx)

def boxlist_ioux(boxlist1, boxlist2):
    time_start = time.time()
    """
    Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    # information for boxlist1 and boxlist2:
    """
    print("##########################################################################")
    print(" length of boxlist1:", len(boxlist1))
    print(" type of boxlist1:", type(boxlist1))
    print(" boxlist1:\n", boxlist1)
    print(" length of boxlist2:", len(boxlist2))
    print(" type of boxlist2:", type(boxlist2))
    print(" boxlist2:\n", boxlist2)
    """
    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    """
    print(" shape of box1:",box1.shape)
    print(" type of box1:", type(box1))
    print(" box1:\n", box1)
    print(" shape of box2:", box2.shape)
    print(" type of box2:", type(box2))
    print(" box2:\n", box2)
    """

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    """
    print(" shape of iou:", iou.shape)
    print(" type of iou:", type(iou))
    print(" iou:\n", iou)
    """
    # Calculation for DIoU #
    ## new way to get diou+
    diou = box_dious(box1, box2)
    dious = diou
    diou = 0.0001 * (1 - diou)

    iou = iou + diou
    """
    print(" shape of diou:", diou.shape)
    print(" type of diou:", type(diou))
    print(" diou:\n", diou)
    print(" shape of iou:", iou.shape)
    print(" type of iou:", type(iou))
    print(" iou:\n", iou)
    print("##########################################################################")
    """
    """
    improvement for IoU ,-->CD-IoU,go to boxlist_ious(boxlist1,boxlist2):
    """
    return iou, dious

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, CDIoU=False, l_CDIoU=False, NCDIoU=False, l_NCDIoU=False, l_NCDIoU_2=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        
    box_1_new = torch.cat((b1_x1, b1_x2, b1_y1, b1_y2), 1)
    box_2_new = torch.cat((b2_x1, b2_x2, b2_y1, b2_y2), 1)
        
    A = torch.sqrt((b1_x1 - b2_x1)**2 + (b1_y1 - b2_y1)**2)
    B = torch.sqrt((b1_x2 - b2_x2)**2 + (b1_y2 - b2_y2)**2)
    C = torch.sqrt((b1_x1 - b2_x1)**2 + (b1_y2 - b2_y2)**2)
    D = torch.sqrt((b1_x2 - b2_x2)**2 + (b1_y1 - b2_y1)**2)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    union_true = w2 * h2 + eps

    iou = inter / union
    iou_true = (inter) / (union_true)
    if GIoU or DIoU or CIoU or CDIoU or l_CDIoU or NCDIoU or l_NCDIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or CDIoU or NCDIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            diou = (A + B + C + D) / 4*torch.sqrt(c2)
            
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif CDIoU:
                return boxlist_ioux(box_1_new, box_2_new)
            elif l_CDIoU:
                return 1 - iou + box_dious(box_1_new, box_2_new)
            elif l_NCDIoU:
                alp = 0.5
                c_area = cw * ch + eps
                return 1 - iou + ((c_area - union) / c_area) + diou
            elif l_NCDIoU_2:
                alp = 0.5
                c_area = cw * ch + eps
                return 2 - iou - iou_true + diou
                
                
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
