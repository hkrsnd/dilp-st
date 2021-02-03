import random
import sys
from plot import plot_line_graph
import pickle
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from clause_generator import ClauseGenerator
from ilp_solver import ILPSolver
from ilp_problem import ILPProblem
from data_utils import DataUtils
sys.path.append('datasets')
sys.path.append('experiments/')
from experiments.eval_utils import get_dataset_with_noise, compute_auc, compute_mse, extract


random.seed(a=7014) # PAPER ID

def normal_experiment(args):
    test_size = 0.3
    du = DataUtils(args.name)
    pos, neg, bk, clauses, lang = du.load_data()
    pos_train, pos_test = train_test_split(
        pos, test_size=test_size, random_state=7014)
    neg_train, neg_test = train_test_split(
        neg, test_size=test_size, random_state=7014)

    pos_train_, neg_train_ = get_dataset_with_noise(pos_train, neg_train, noise_rate=args.noise_rate)


    if args.name == 'member':
        beam_step = 3
        N_beam = 3
    elif args.name == 'subtree':
        beam_step = 3
        N_beam = 15
    else:
        beam_step = 5
        N_beam = 10
    N_max = 50
    N = 1

    ilp_train = ILPProblem(pos_train_, neg_train_, bk, lang, name=args.name)
    ilp_train.print()
    CG = ClauseGenerator(ilp_train, infer_step=args.T, max_depth=1, max_body_len=1)
    solver = ILPSolver(ilp_train, C_0=clauses, CG=CG, m=args.m, infer_step=args.T)
    clauses_, Ws_list, loss_list_list = solver.train_N(
        N=N, gen_mode='beam', N_max=N_max, T_beam=beam_step, N_beam=N_beam, epoch=args.epoch, lr=args.lr, wd=0.0)
    v_list, facts = solver.predict_N(pos_test, neg_test, clauses_, Ws_list)
    mse = compute_mse(pos_test, neg_test, v_list[0], facts)
    auc = compute_auc(pos_test, neg_test, v_list[0], facts)

    print('====== TEST SCORE =======')
    print('Mean-squared test error: ', mse)
    print('AUC: ', auc)


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
