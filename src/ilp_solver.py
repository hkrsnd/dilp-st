import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from fact_enumerator import FactEnumerator as FE
from ilp_problem import ILPProblem
from infer import InferModule, InferModulePair, convert
from optimizer import WeightOptimizer
from tensor_encoder import TensorEncoder as TE
import torch.nn.functional as F

random.seed(a=7014)  # PAPER ID 7014

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ILPSolver():
    """
    differentiable inductive logic programming solver

    Parameters
    ----------
    Q : (Set[.logic.Atom], Set[.logic.Atom], Set[.logic.Atom], Set[.logic.Atom])
        inductive logic programming problem (P, N, B, L)
    C_0 : List[.logic.Clause]
        set of initial clauses from the clause-generation step
    CG : .clause_generator.ClauseGenerator
        clause generator
    m : int
        size of logic program of the solution
    infer_step : int
        number of steps of forward-chaining inference
    """

    def __init__(self, Q, C_0, CG, m=3, infer_step=5, im_mode='softor'):
        self.Q = Q
        self.bk = Q.bk
        self.lang = Q.lang
        self.C_0 = C_0
        self.CG = CG
        self.m = m
        self.infer_step = infer_step
        self.im_mode = im_mode
        self.labels = torch.cat([
            torch.ones((len(self.Q.pos), )).to(device),
            torch.zeros((len(self.Q.neg), )).to(device)
        ], dim=0).to(device)

    def get_train_idxs(self, facts):
        """
        get indexes of training data

        Inputs
        ------
        facts : List[.logic.Atom]
            set of ground atoms
        """
        pos_train_idxs = torch.tensor(
            [facts.index(e) for e in self.Q.pos]).to(device)
        neg_train_idxs = torch.tensor(
            [facts.index(e) for e in self.Q.neg]).to(device)
        return torch.cat([pos_train_idxs, neg_train_idxs], dim=0)

    # def count_params(self):
    #    return self.N_params

    def init_valuation(self, facts):
        """
        initialize valuations

        Inputs
        ------
        facts : List[.logic.Atom]
            set of ground atoms      

        Returns
        -------
        v : torch.tensor(|facts|, )
            valuation vector
        """
        ls = [
            1.0 if (fact in self.Q.bk)
            else 0.0 for fact in facts]
        ls[1] = 1.0  # TRUE ATOM
        return torch.tensor(ls).to(device)

    def print_program(self, C, IM):
        """
        print summary of logic programs by discretizing continuous weights

        Inputs
        ------
        C : List[.logic.Clause]
            set of clauses
        IM : .logic.infer.InferModule
            infer module that contains clause weights
        """
        print('====== LEARNED PROGRAM ======')
        if self.im_mode == 'softor':
            for i, W in enumerate(IM.Ws):
                W_ = IM.softmax(W)
                max_i = np.argmax(W_.detach().cpu().numpy())
                print('C_'+str(i)+': ',
                      C[max_i], W_[max_i].detach().cpu().item())
        else:
            W = F.softmax(IM.Ws[0], dim=0).detach().cpu().numpy()
            max_i, max_j = np.unravel_index(np.argmax(W), W.shape)
            print(W[max_i][max_j])
            print(C[max_i])
            print(C[max_j])

    def train(self, gen_mode='beam', N_max=100, T_beam=7, N_beam=20, epoch=5000, lr=1e-2, wd=0.0):
        """
        training the model
        1. generate clauses
        2. enumerate facts
        3. build index tensors
        4. optimize weights by gradient descent

        Inputs
        ------
        gen_mode : str
            clause generation mode
            'beam' - generate with beam searching
            'naive' - generate without beam searching
        N_max : int
            max number of clauses to be generated
        T_beam : int
            number of steps for beam searching
        N_beam : int
            size of beam 
        epoch : int
            number of epochs for gradient descent optimization
        lr : float
            learning rate for gradient descent optimization
        wd : float
            weight decay for gradient descent optimization

        Returns
        -------
        clauses : List[.logic.Clause]
            generated clauses
        Ws : List[torch.Tensor((|clauses|, ))]
            set of clause weights
        loss_list : List[float]
            list of loss values
        """
        # Generate clauses
        print("Generating clauses...")
        clauses = self.CG.generate(
            self.C_0, gen_mode=gen_mode, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
        # Enumerate facts
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            self.Q, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        if self.im_mode == 'softor':
            IM = InferModule(X, self.m, v_0, self.infer_step)
            self.N_params = len(X) * self.m
        elif self.im_mode == 'pair':
            IM = InferModulePair(X, self.m, v_0, self.infer_step)
            self.N_params = len(X) * len(X)

        # Optimize weights
        train_idxs = self.get_train_idxs(facts)
        optim = WeightOptimizer(IM, train_idxs, self.labels, lr=lr, wd=wd)
        print("Learning weights...")
        IM_, loss_list = optim.optimize_weights(epoch=epoch)
        self.print_program(clauses, IM_)
        return clauses, IM_.Ws, loss_list

    def train_time(self, gen_mode='beam', N_max=100, T_beam=7, N_beam=20, epoch=5000, lr=1e-2, wd=0.0):
        """
        training the model with measuring the time of each step of the optimization
        1. generate clauses
        2. enumerate facts
        3. build index tensors
        4. optimize weights by gradient descent

        Inputs
        ------
        gen_mode : str
            clause generation mode
            'beam' - generate with beam searching
            'naive' - generate without beam searching
        N_max : int
            max number of clauses to be generated
        T_beam : int
            number of steps for beam searching
        N_beam : int
            size of beam 
        epoch : int
            number of epochs for gradient descent optimization
        lr : float
            learning rate for gradient descent optimization
        wd : float
            weight decay for gradient descent optimization

        Returns
        -------
        clauses : List[.logic.Clause]
            generated clauses
        Ws : List[torch.Tensor((|clauses|, ))]
            set of clause weights
        loss_list : List[float]
            list of loss values            
        """
        # Generate clausespl
        print("Generating clauses...")
        clauses = self.CG.generate(
            self.C_0, gen_mode=gen_mode, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
        # Enumerate facts
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            self.Q, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        if self.im_mode == 'softor':
            IM = InferModule(X, self.m, v_0, self.infer_step)
            self.N_params = len(X) * self.m
        elif self.im_mode == 'pair':
            IM = InferModulePair(X, self.m, v_0, self.infer_step)
            self.N_params = len(X) * len(X)

        # Optimize weights
        train_idxs = self.get_train_idxs(facts)
        optim = WeightOptimizer(IM, train_idxs, self.labels, lr=lr, wd=wd)
        IM_, loss_list, times = optim.optimize_weights_time(epoch=epoch)
        self.print_program(clauses, IM_)
        return clauses, IM_.Ws, loss_list, times

    def train_N(self, N=10, gen_mode='beam', N_max=100, T_beam=7, N_beam=20, epoch=5000, lr=1e-2, wd=0.0):
        """
        training the model with multiple traials to evaluate mean and variance of results
        1. generate clauses
        2. enumerate facts
        3. build index tensors
        4. optimize weights by gradient descent

        Inputs
        ------
        N : int
            number of trials of training
        gen_mode : str
            clause generation mode
            'beam' - generate with beam searching
            'naive' - generate without beam searching
        N_max : int
            max number of clauses to be generated
        T_beam : int
            number of steps for beam searching
        N_beam : int
            size of beam 
        epoch : int
            number of epochs for gradient descent optimization
        lr : float
            learning rate for gradient descent optimization
        wd : float
            weight decay for gradient descent optimization

        Returns
        -------
        clauses : List[.logic.Clause]
            generated clauses
        Ws_list : List[List[torch.Tensor((|clauses|, ))]]
            list of set of learned clause weights
        loss_list_list : List[List[float]]
            lisf of list of loss values            
        """
        # Generate clausespl
        print("Beam-searching for clauses...")
        clauses = self.CG.generate(
            self.C_0, gen_mode=gen_mode, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
        # Enumerate facts
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            self.Q, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        self.N_params = len(X) * self.m
        # Optimize weights
        train_idxs = self.get_train_idxs(facts)
        Ws_list = []
        loss_list_list = []
        # multiple trials to evaluate mean and variance of results
        for ni in range(N):
            IM = InferModule(X, self.m, v_0, self.infer_step)
            optim = WeightOptimizer(IM, train_idxs, self.labels, lr=lr, wd=wd)
            IM_, loss_list = optim.optimize_weights(epoch=epoch)
            #self.print_program(clauses, IM_)
            Ws_list.append(IM_.Ws)
            loss_list_list.append(loss_list)
        return clauses, Ws_list, loss_list_list

    def predict_N(self, pos_test, neg_test, clauses, Ws_list):
        """
        predict for test data with multiple traials to evaluate mean and variance of results

        Inputs
        ------
        pos_test : List[.logic.Atom]
            test positive examples
        neg_test : List[.logic.Atom]
            test negative examples
        clauses : List[.logic.Clause]
            generated clauses
        Ws_list : List[List[torch.Tensor((|clauses|, ))]]
            list of weights for clauses yielded by multiple trials 

        Returns
        -------
        v_list : List[torch.Tensor(|facts|, )]
            list of valuation vectors 
            v_0, v_1,..., v_T
        facts : List[.logic.Atom]
            enumerated ground atoms
        """
        print('Prediction for test data...')
        Q_test = ILPProblem(pos_test, neg_test, self.bk, self.lang)
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            Q_test, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        IM = InferModule(X, self.m, v_0, self.infer_step)
        v_list = []

        for Ws in Ws_list:
            IM.Ws = Ws
            v_list.append(IM.infer())
            self.print_program(clauses, IM)
        return v_list, facts

    def predict(self, pos_test, neg_test, clauses, Ws):
        """
        predict for test data with multiple traials to evaluate mean and variance of results

        Inputs
        ------
        pos_test : List[.logic.Atom]
            test positive examples
        neg_test : List[.logic.Atom]
            test negative examples
        clauses : List[.logic.Clause]
            generated clauses
        Ws : List[List[torch.Tensor(|clauses|, )]]
            list of weights for clauses

        Returns
        -------
        v_list : torch.Tensor(|facts|, )
            valuation vector 
            v_T
        facts : List[.logic.Atom]
            enumerated ground atoms
        """
        Q_test = ILPProblem(pos_test, neg_test, self.bk, self.lang)
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            Q_test, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        if self.im_mode == 'softor':
            IM = InferModule(X, self.m, v_0, self.infer_step)
        elif self.im_mode == 'pair':
            IM = InferModulePair(X, self.m, v_0, self.infer_step)
        IM.Ws = Ws
        return IM.infer(), facts

    def get_valuation_memory(self, pos_test, neg_test, clauses, Ws):
        """
        predict for test data with multiple traials to evaluate mean and variance of results

        Inputs
        ------
        pos_test : List[.logic.Atom]
            test positive examples
        neg_test : List[.logic.Atom]
            test negative examples
        clauses : List[.logic.Clause]
            generated clauses
        Ws : List[List[torch.Tensor(|clauses|, )]]
            list of weights for clauses

        Returns
        -------
        valuation memory : torch.Tensor(|facts|, )
            list of valuation vectors 
            v_0, v_1,..., v_T
        facts : List[.logic.Atom]
            enumerated ground atoms
        """
        Q_test = ILPProblem(pos_test, neg_test, self.bk, self.lang)
        print("Enumerating facts...")
        facts = FE.enumerate_facts(
            Q_test, clauses, infer_step=self.infer_step)
        # Build tensors
        print("Building tensors...")
        TEncoder = TE(facts, clauses, self.infer_step)
        X = TEncoder.encode()
        # Build Infer module
        v_0 = self.init_valuation(facts)
        if self.im_mode == 'softor':
            IM = InferModule(X, self.m, v_0, self.infer_step)
        elif self.im_mode == 'pair':
            IM = InferModulePair(X, self.m, v_0, self.infer_step)
        IM.Ws = Ws
        v_T = IM.infer()
        return IM.valuation_memory, facts

    def count_params(self):
        return self.N_params
