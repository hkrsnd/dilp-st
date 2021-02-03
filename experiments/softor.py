from plot import plot_loss_compare
from ilp_solver import ILPSolver
from ilp_problem import ILPProblem
from clause_generator import ClauseGenerator
from data_utils import DataUtils
import torch.nn.functional as F
from torch.distributions import Categorical  # for entropy
import torch as torch
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from statistics import mean, median, variance, stdev
from math import log, e
import sys
sys.path.append('datasets')
sys.path.append('src/')
sys.path.append('experiments/')


def softmax2d(W):
    return F.softmax(torch.flatten(W), dim=0).view(-1, len(W))


def softor_experiment(args, max_n=5, test_size=0.3):
    du = DataUtils(args.name)
    pos, neg, bk, clauses, lang = du.load_data()
    pos_train, pos_test = train_test_split(
        pos, test_size=test_size, random_state=7014)
    neg_train, neg_test = train_test_split(
        neg, test_size=test_size, random_state=7014)

    ilp_train = ILPProblem(pos_train, neg_train, bk, lang, name=args.name)

    N_max = 50

    if args.name in ['member']:
        T_beam = 3
        N_beam = 3
        m = 3

    elif args.name in ['subtree']:
        T_beam = 3
        N_beam = 15
        m = 4

    else:
        T_beam = 5
        N_beam = 10
        m = 3

    CG = ClauseGenerator(ilp_train, infer_step=args.T,
                         max_depth=1, max_body_len=1)
    solver = ILPSolver(ilp_train, C_0=clauses, CG=CG,
                       m=args.m, infer_step=args.T)
    #solver = ILPSolver(ilp_train, C_0=clauses, m=args.m, infer_step=args.T, im_mode='softmax')
    clauses_, Ws_, loss_list, times = solver.train_time(
        gen_mode='beam', N_max=N_max, T_beam=T_beam, N_beam=N_beam, epoch=args.epoch, lr=args.lr, wd=0.0)
    print('Ws: ')
    for W in Ws_:
        print(F.softmax(W, dim=0))
    v_, facts = solver.predict(pos_test, neg_test, clauses_, Ws_)
    auc = compute_auc(pos_test, neg_test, v_, facts)
    mse = compute_mse(pos_test, neg_test, v_, facts)
    ent = compute_ent(Ws_, gen_mode='softmax')
    print('ENT:', ent)

    print('AUC:', auc)

    df = {}
    df['AUC'] = auc
    df['N_params'] = solver.count_params()
    df['time'] = mean(times)
    df['std'] = stdev(times)
    df['MSE'] = mse
    df['ENT'] = compute_ent(Ws_, gen_mode='softmax')

    path = 'results/'+args.name + '_softor' + '.txt'
    save(path, df)

    # PAIR
    CG = ClauseGenerator(ilp_train, infer_step=args.T,
                         max_depth=1, max_body_len=1)
    solver = ILPSolver(ilp_train, C_0=clauses, CG=CG,
                       m=args.m, infer_step=args.T, im_mode='pair')
    #solver = ILPSolver(ilp_train, C_0=clauses, m=2, infer_step=args.T, im_mode='pair')
    clauses_, Ws_, pair_loss_list, times = solver.train_time(
        gen_mode='beam', N_max=N_max, T_beam=T_beam, N_beam=N_beam, epoch=args.epoch, lr=args.lr, wd=0.0)
    print('Ws: ')
    print(softmax2d(Ws_[0]))
    v_, facts = solver.predict(pos_test, neg_test, clauses_, Ws_)
    auc = compute_auc(pos_test, neg_test, v_, facts)
    mse = compute_mse(pos_test, neg_test, v_, facts)

    df = {}
    df['AUC_pair'] = auc
    df['N_params_pair'] = solver.count_params()
    df['time_pair'] = mean(times)
    df['std_pair'] = stdev(times)
    df['MSE_pair'] = mse
    df['ENT_pair'] = compute_ent(Ws_, gen_mode='pair')

    path = 'results/'+args.name+'_pair' + '.txt'
    save(path, df)
    print(df)

    loss_path = 'imgs/softor/loss/' + args.name + '.pdf'
    ys_list = [loss_list, pair_loss_list]
    plot_loss_compare(loss_path, ys_list, args.name)


def save(path, my_dict):
    with open(path, 'w') as f:
        for key in my_dict.keys():
            f.write("%s,%s\n" % (key, my_dict[key]))


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


def compute_ent(Ws, gen_mode='softmax'):
    if gen_mode == 'softmax':
        ent_sum = 0
        for W in Ws:
            ent_sum = vector_ent(F.softmax(W, dim=0).cpu()).item()
        return ent_sum / len(Ws)
    else:
        #W = F.softmax(torch.flatten(Ws[0]), dim=0).view(-1, len(Ws[0]))
        W = softmax2d(Ws[0])
        return matrix_ent(W.cpu()).item()


def vector_ent(v):
    ent = torch.tensor([0.0])
    base = e
    for i in range(len(v)):
        if 0 < v[i] < 1:
            ent -= v[i] * log(v[i], e)
    return ent


def matrix_ent(M):
    ent = torch.tensor([0.0])
    base = e
    v = torch.flatten(M)
    for i in range(len(v)):
        if 0 < v[i] < 1:
            ent -= v[i] * log(v[i], e)
    return ent
