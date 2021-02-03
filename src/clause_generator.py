import torch
from refinement import RefinementGenerator
from infer import InferModule, convert
from fact_enumerator import FactEnumerator as FE
from logic_ops import is_entailed
from tensor_encoder import TensorEncoder as TE

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ClauseGenerator():
    """
    clause generator by refinement and beam search

    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, ilp_problem, infer_step, max_depth=1, max_body_len=1):
        self.ilp_problem = ilp_problem
        self.infer_step = infer_step
        self.rgen = RefinementGenerator(
            lang=ilp_problem.lang, max_depth=max_depth, max_body_len=max_body_len)
        self.max_depth = max_depth
        self.max_body_len = max_body_len
        self.bce_loss = torch.nn.BCELoss()
        self.labels = torch.cat([
            torch.ones((len(self.ilp_problem.pos), )),
        ], dim=0).to(device)

    def generate(self, C_0, gen_mode='beam', T_beam=7, N_beam=20, N_max=100):
        """
        call clause generation function with or without beam-searching

        Inputs
        ------
        C_0 : Set[.logic.Clause]
            a set of initial clauses
        gen_mode : string
            a generation mode
            'beam' - with beam-searching
            'naive' - without beam-searching
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated

        Returns
        -------
        C : Set[.logic.Clause]
            set of generated clauses
        """
        if gen_mode == 'beam':
            return self.beam_search(C_0, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
        elif gen_mode == 'naive':
            return self.naive(C_0, T_beam=T_beam, N_max=N_max)

    def beam_search_clause(self, clause, T_beam=7, N_beam=20, N_max=100):
        """
        perform beam-searching from a clause

        Inputs
        ------
        clause : Clause
            initial clause
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated

        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses
        """
        step = 0
        init_step = 0
        B = [clause]
        C = set()
        C_dic = {}
        B_ = []
        lang = self.ilp_problem.lang

        while step < T_beam:
            #print('Beam step: ', str(step),  'Beam: ', len(B))
            B_new = {}
            for c in B:
                refs = self.rgen.refinement_clauses([c])
                # remove already appeared refs
                refs = list(set(refs).difference(set(B_)))
                B_.extend(refs)
                for ref in refs:
                    loss = self.eval_clause(ref)
                    B_new[ref] = loss
                    C_dic[ref] = loss
                C = C.union(set([c]))
                if len(C) >= N_max:
                    break
            B_new_sorted = sorted(B_new.items(), key=lambda x: x[1])
            # top N_beam refiements
            B_new_sorted = B_new_sorted[:N_beam]
            B = [x[0] for x in B_new_sorted]
            step += 1
            if len(B) == 0:
                break
            if len(C) >= N_max:
                break
        return C

    def beam_search(self, C_0, T_beam=7, N_beam=20, N_max=100):
        """
        generate clauses by beam-searching from initial clauses

        Inputs
        ------
        C_0 : Set[.logic.Clause]
            set of initial clauses
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated

        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses        
        """
        C = set()
        for clause in C_0:
            C = C.union(self.beam_search_clause(
                clause, T_beam, N_beam, N_max))
        C = sorted(list(C))
        print('======= BEAM SEARCHED CLAUSES ======')
        for c in C:
            print(c)
        return C

    def naive(self, C_0, T_beam=7, N_max=100):
        """
        Generate clauses without beam-searching from clauses.

        Inputs
        ------
        C_0 : Set[.logic.Clause]
            set of initial clauses
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated

        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses                
        """
        step = 0
        C = set()
        C_next = set(C_0)
        while step < T_beam:
            for c in C_next:
                refs = self.rgen.refinement_clauses([c])
                C = C.union(set([c]))
                C_next = C_next.difference(set([c]))
                C_next = C_next.union(set(refs))
                if len(C) >= N_max:
                    break
            if len(C) >= N_max:
                break
        print('======= GENERATED CLAUSES ======')
        for c in C:
            print(c)
        return list(C)

    def eval_clause_by_entailment(self, clause):
        """
        evaluate a clause with examples by computing emtailments
        slower than using differentiable implementation

        Inputs
        ------
        clause : Clause
            clause to be evaluated

        Returns
        -------
        score : float
            evaluation score
            computed as a loss with the input clause
        """
        score = 0
        for e in self.ilp_problem.pos:
            if is_entailed(e, clause, self.ilp_problem.bk, self.infer_step):
                score += 1
        print("clause: ", clause,  "score: ", score)
        return -score               

    def eval_clause(self, clause):
        """
        evaluate a clause with examples using differentiable implementation
        faster than computing entailment exactly

        Inputs
        ------
        clause : Clause
            clause to be evaluated

        Returns
        -------
        score : float
            evaluation score
            computed as a loss with the input clause
        """
        facts = FE.enumerate_facts(self.ilp_problem, C=[clause], infer_step=self.infer_step)
        v_0 = self.get_v_0(clause, self.ilp_problem, facts)
        # Build tensors using only input clause
        TEncoder = TE(facts, clauses=[clause], infer_step=self.infer_step)
        X = TEncoder.encode()
        IM = InferModule(X, 1, v_0, infer_step=self.infer_step)
        valuation = IM.infer()
        probs = torch.tensor([valuation[facts.index(p)]
                              for p in self.ilp_problem.pos]).to(device)
        score = self.bce_loss(probs, self.labels)
        return score

    def get_v_0(self, clause, ilp_problem, facts):
        """
        evaluate a clause with examples

        Inputs
        ------
        clause : Clause
            clause to be evaluated
        ilp_problem : ILPProblem
        facts : Set[Atom]

        Returns
        -------
        v_0 : (|facts|, ) torch.tensor
            1-0 vector representing background knowledge in the ilp problem
        """
        head_pred = clause.head.pred
        pos_diff = [atom for atom in ilp_problem.pos if atom.pred != head_pred]
        B = ilp_problem.bk + pos_diff
        return convert(B, facts)
