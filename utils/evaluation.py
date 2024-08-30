import os
import sys
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score


def find_nearest(array, value):

    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def acc_at_t(preds, labels, t):

    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    return acc_at_95


def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    return aupr


def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True  Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def compute_aurc(preds, labels, confidence):
    coverages = []
    risks = []
    weights = []
    tmp_weight = 0

    correct = (preds == labels)
    residuals = 1 - correct

    n = len(residuals)
    cov = n

    idx_sorted = np.argsort(confidence)
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov / n)
    risks.append(error_sum / n)
    
    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum /(n - 1 - i)
        tmp_weight += 1
        
        if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    aurc = sum([(risks[i] + risks[i+1]) * 0.5  * weights[i] for i in range(len(weights))])

    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err + np.finfo(err.dtype).eps))
    e_aurc = aurc - kappa_star_aurc

    return aurc, e_aurc

def compute_oaa(mixed_scores_arr, mixed_preds_arr, mixed_labels_arr, t, in_length):
    in_scores, ood_scores = mixed_scores_arr[:in_length], mixed_scores_arr[in_length:]
    in_preds, ood_preds = mixed_preds_arr[:in_length], mixed_preds_arr[in_length:]
    in_labels_arr, ood_labels_arr = mixed_labels_arr[:in_length], mixed_labels_arr[in_length:]

    # predicted ID
    TN_index = in_scores <= t
    FN_index = ood_scores <= t
    TN_preds = in_preds[TN_index]
    FN_preds = ood_preds[FN_index]
    TN_labels = in_labels_arr[TN_index]
    FN_labels = ood_labels_arr[FN_index]
    rob_correct = (TN_preds == TN_labels).sum() + (FN_preds == FN_labels).sum()

    # predicted OOD
    TP_index = ood_scores > t
    FP_index = in_scores > t 
    TP_preds = ood_preds[TP_index]
    FP_preds = in_preds[FP_index]
    TP_labels = ood_labels_arr[TP_index]
    FP_labels = in_labels_arr[FP_index]
    det_correct = (TP_preds != TP_labels).sum() + (FP_preds != FP_labels).sum()

    rob_cor_rate = rob_correct / (len(mixed_scores_arr))
    det_cor_rate = det_correct / (len(mixed_scores_arr))

    return rob_cor_rate, det_cor_rate

def metric(closed_preds, closed_labels, open_preds, open_labels, mtypes=['AUROC']):
    results = dict()

    fpr, tpr, thresh = roc_curve(open_labels, open_preds, drop_intermediate=False)

    for mtype in mtypes:
        if mtype == 'TPR':
            TPR = acc_at_95_tpr(open_preds, open_labels, thresh, tpr)
            results['TPR'] = round(TPR, 4)
        elif mtype == 'AUROC':
            AUROC = compute_auroc(open_preds, open_labels)
            results['AUROC'] = round(AUROC, 4)
        elif mtype == 'AUPR':
            AUPR = compute_aupr(open_preds, open_labels, normalised_ap=False)
            results['AUPR'] = round(AUPR, 4)
        elif mtype == 'AURC':
            closed_set_preds_pred_cls = np.array(closed_preds[0]).argmax(axis=-1)
            labels_known_cls = np.array(closed_labels[0])
            AURC, eAURC = compute_aurc(closed_set_preds_pred_cls, labels_known_cls, open_preds)
            results['AURC'] = round(AURC, 4)
            results['eAURC'] = round(eAURC, 4)
        elif mtype == 'OSCR':
            open_set_preds_known_cls = open_preds[~open_labels.astype('bool')]
            open_set_preds_unknown_cls = open_preds[open_labels.astype('bool')]
            closed_set_preds_pred_cls = np.array(closed_preds[0]).argmax(axis=-1)
            labels_known_cls = np.array(closed_labels[0])
            if len(open_set_preds_known_cls) == len(closed_set_preds_pred_cls):
                OSCR = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)
                results['OSCR'] = round(OSCR, 4)
            
    return results
