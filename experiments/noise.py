import copy
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from plot import plot_line_graph, plot_line_graph_baseline, plot_loss, plot_line_graph_err, plot_line_graph_baseline_err
from clause_generator import ClauseGenerator
from ilp_solver import ILPSolver
from ilp_problem import ILPProblem
from data_utils import DataUtils
import sys
sys.path.append('datasets')
sys.path.append('src/')
sys.path.append('experiments/')


seed = 7014  # PAPER ID 7014
random.seed(a=seed)


def extract(valuation, facts, fact):
    return valuation[facts.index(fact)].detach().cpu().item()


def noise_experiment(args, test_size=0.3):
    du = DataUtils(args.name)
    pos, neg, bk, clauses, lang = du.load_data()
    pos_train, pos_test = train_test_split(
        pos, test_size=test_size, random_state=seed)
    neg_train, neg_test = train_test_split(
        neg, test_size=test_size, random_state=seed)

    noise_rates = [0.0, 0.05, 0.1, 0.15, 0.2,
                   0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    baseline_auc = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    datasets = get_datasets_with_noise(pos_train, neg_train, noise_rates)
    AUCs = []
    AUC_stds = []
    MSEs = []
    MSE_stds = []
    N = 5  # how many times to perform weight learn

    if args.name == 'member':
        T_beam = 3
        N_beam = 3

    elif args.name == 'subtree':
        T_beam = 3
        N_beam = 15
    else:
        T_beam = 5
        N_beam = 10

    N_max = 50

    for i, (pos_train, neg_train) in enumerate(datasets):
        ilp_train = ILPProblem(pos_train, neg_train, bk, lang, name=args.name)
        print('NOISE RATE: ', noise_rates[i])
        ilp_train.print()
        CG = ClauseGenerator(ilp_train, infer_step=args.T,
                             max_depth=1, max_body_len=1)
        solver = ILPSolver(ilp_train, C_0=clauses, CG=CG,
                           m=args.m, infer_step=args.T)
        clauses_, Ws_list, loss_list_list = solver.train_N(
            N=N, gen_mode='beam', N_max=N_max, T_beam=T_beam, N_beam=N_beam, epoch=args.epoch, lr=args.lr, wd=0.0)
        v_list, facts = solver.predict_N(pos_test, neg_test, clauses_, Ws_list)

        auc_list = np.array(
            [compute_auc(pos_test, neg_test, v_, facts) for v_ in v_list])
        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)
        AUCs.append(auc_mean)
        AUC_stds.append(auc_std)

        mse_list = np.array(
            [compute_mse(pos_test, neg_test, v_, facts) for v_ in v_list])
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)
        MSEs.append(mse_mean)
        MSE_stds.append(mse_std)
        for j in range(N):
            loss_path = 'imgs/noise/loss/' + args.name + \
                '[noise:' + str(noise_rates[i]) + ']-' + str(j) + '.pdf'
            plot_loss(loss_path, loss_list_list[j], args.name +
                      ':[noise:' + str(noise_rates[i]) + ']-' + str(j))

    # plot AUC with baseline
    path_auc = 'imgs/noise/' + args.name + '_AUC.pdf'
    path_mse = 'imgs/noise/' + args.name + '_MSE.pdf'

    print(AUC_stds)
    print(MSE_stds)

    plot_line_graph_baseline_err(path=path_auc, xs=noise_rates, ys=AUCs, err=AUC_stds,
                                 xlabel='Proportion of mislabeled training data', ylabel='AUC', title=args.name, baseline=baseline_auc)
    # plot MSR with std
    plot_line_graph_err(path=path_mse, xs=noise_rates, ys=MSEs, err=MSE_stds,
                        xlabel='Proportion of mislabeled training data', ylabel='Mean-squared test error', title=args.name)


def get_datasets_with_noise(pos_train, neg_train, noise_rates):
    datasets = []
    pos_noise_indexes = random.sample(range(len(pos_train)), len(pos_train))
    neg_noise_indexes = random.sample(range(len(neg_train)), len(neg_train))

    pos_batch = int(len(pos_train) / 20)  # assume 5% stride of noise
    neg_batch = int(len(neg_train) / 20)  # assume 5% stride of noise

    for i in range(len(noise_rates)):
        pos_, neg_ = exchange_by_index(
            pos_train, neg_train, pos_noise_indexes[:i*pos_batch], neg_noise_indexes[:i*neg_batch])
        datasets.append((pos_, neg_))
    return datasets


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
