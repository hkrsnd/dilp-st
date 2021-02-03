from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from plot import plot_line_graph_compare_err, plot_loss_compare
from ilp_solver import ILPSolver
from ilp_problem import ILPProblem
from clause_generator import ClauseGenerator
from data_utils import DataUtils
import numpy as np
import sys
sys.path.append('datasets')
sys.path.append('src/')
sys.path.append('experiments/')

seed = 7014  # PAPER ID 7104


def step_experiment(args, max_n=5, test_size=0.3):
    du = DataUtils(args.name)
    pos, neg, bk, clauses, lang = du.load_data()
    pos_train, pos_test = train_test_split(
        pos, test_size=test_size, random_state=seed)
    neg_train, neg_test = train_test_split(
        neg, test_size=test_size, random_state=seed)

    ilp_train = ILPProblem(pos_train, neg_train, bk, lang, name=args.name)
    if args.name in ['member']:
        N_max_list = [3, 6, 9, 12]
    else:
        N_max_list = [10, 15, 20, 25, 30, 35, 40]

    if args.name in ['subtree']:
        N_beam = 15
        T_beam = 3
    else:
        N_beam = 10
        T_beam = 7

    AUCs = []
    AUC_stds = []
    MSEs = []
    MSE_stds = []
    N = 5  # how many times to perform weight learn

    naive_AUCs = []
    naive_AUC_stds = []
    naive_MSEs = []
    naive_MSE_stds = []

    for N_max in N_max_list:
        CG = ClauseGenerator(ilp_train, infer_step=args.T,
                             max_depth=1, max_body_len=1)
        solver = ILPSolver(ilp_train, C_0=clauses, CG=CG,
                           m=args.m, infer_step=args.T)
        #solver = ILPSolver(ilp_train, C_0=clauses, m=args.m, infer_step=args.T)
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

        # NAIVE
        CG = ClauseGenerator(ilp_train, infer_step=args.T,
                             max_depth=1, max_body_len=1)
        solver = ILPSolver(ilp_train, C_0=clauses, CG=CG,
                           m=args.m, infer_step=args.T)
        clauses_, Ws_list, naive_loss_list_list = solver.train_N(
            N=N, gen_mode='naive', N_max=N_max, T_beam=T_beam, N_beam=N_beam, epoch=args.epoch, lr=args.lr, wd=0.0)
        v_list, facts = solver.predict_N(pos_test, neg_test, clauses_, Ws_list)

        auc_list = np.array(
            [compute_auc(pos_test, neg_test, v_, facts) for v_ in v_list])
        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)
        naive_AUCs.append(auc_mean)
        naive_AUC_stds.append(auc_std)

        mse_list = np.array(
            [compute_mse(pos_test, neg_test, v_, facts) for v_ in v_list])
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)
        naive_MSEs.append(mse_mean)
        naive_MSE_stds.append(mse_std)

        for j in range(N):
            loss_path = 'imgs/step/loss/' + args.name + \
                '[N_max:' + str(N_max) + ']-' + str(j) + '.pdf'
            ys_list = [loss_list_list[j], naive_loss_list_list[j]]
            plot_loss_compare(loss_path, ys_list, args.name +
                              ':[N_max:' + str(N_max) + ']-' + str(j))

    path_auc = 'imgs/step/' + args.name + '_AUC.pdf'
    path_mse = 'imgs/step/' + args.name + '_MSE.pdf'
    print(AUC_stds)
    print(MSE_stds)
    labels = ['proposed', 'naive']

    plot_line_graph_compare_err(path=path_auc, xs=N_max_list, ys_list=[AUCs, naive_AUCs], err_list=[AUC_stds, naive_AUC_stds],
                                xlabel='Number of clauses', ylabel='AUC', title=args.name, labels=labels)
    plot_line_graph_compare_err(path=path_mse, xs=N_max_list, ys_list=[MSEs, naive_MSEs], err_list=[MSE_stds, naive_MSE_stds],
                                xlabel='Number of clauses', ylabel='Mean-squared test error', title=args.name, labels=labels)


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
