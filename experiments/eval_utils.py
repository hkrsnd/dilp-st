import copy
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error


def extract(valuation, facts, fact):
    return valuation[facts.index(fact)].detach().cpu().item()


def compute_auc(pos, neg, valuation, facts):
    scores = []
    labels = []
    for e in pos:
        p = extract(valuation, facts, e)
        scores.append(p)
        labels.append(1)
    for e in neg:
        p = extract(valuation, facts, e)
        scores.append(p)
        labels.append(0)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return auc(fpr, tpr)


def compute_mse(pos, neg, valuation, facts):
    scores = []
    labels = []
    for e in pos:
        p = extract(valuation, facts, e)
        scores.append(p)
        labels.append(1)
    for e in neg:
        p = extract(valuation, facts, e)
        scores.append(p)
        labels.append(0)
    return mean_squared_error(labels, scores)


def get_dataset_with_noise(pos_train, neg_train, noise_rate):
    N = int(len(pos_train) * noise_rate)
    pos_noise_index = random.sample(range(len(pos_train)), N)
    neg_noise_index = random.sample(range(len(neg_train)), N)
    pos_, neg_ = exchange_by_index(
        pos_train, neg_train, pos_noise_index, neg_noise_index)
    return pos_, neg_


def exchange_by_index(pos_train, neg_train, pos_noise_indexes, neg_noise_indexes):
    pos_result = copy.deepcopy(pos_train)
    neg_result = copy.deepcopy(neg_train)
    pos_to_add = np.array(pos_train)[pos_noise_indexes]
    neg_to_add = np.array(neg_train)[neg_noise_indexes]

    for e in pos_to_add:
        neg_result.append(e)
        pos_result.remove(e)
    for e in neg_to_add:
        pos_result.append(e)
        neg_result.remove(e)

    return pos_result, neg_result
