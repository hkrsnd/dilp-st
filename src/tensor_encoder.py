import torch
from logic_ops import unify, subs_list

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class TensorEncoder():
    """
    tensor encoder

    Parameters
    ----------
    facts : List[.logic.Atom]
        enumerated ground atoms
    clauses : List[.logic.Clause]
        generated clauses
    infer_step : int
        number of steps in the forward-chaining inference
    """

    def __init__(self, facts, clauses, infer_step):
        self.facts = facts
        self.clauses = clauses
        self.max_body_len = max([len(clause.body)
                                 for clause in clauses] + [1])
        self.infer_step = infer_step
        self.head_unifier_dic = self.build_head_unifier_dic()
        self.fact_index_dic = self.build_fact_index_dic()

    def encode(self):
        """
        compute index tensors for the differentiable inference

        Returns
        -------
        X : torch.tensor((|facts|,|clauses|,|max_body_len|, ))
            index tensor
        """
        X = torch.zeros(
            (len(self.clauses), len(self.facts), self.max_body_len), dtype=torch.long).to(device)
        for ci, clause in enumerate(self.clauses):
            X_c = self.build_X_c(clause)
            X[ci, :, :] = X_c
        return X

    def build_fact_index_dic(self):
        """
        build dictionary [FACT -> INDEX]

        Returns
        -------
        dic : {.logic.Atom -> int}
            dictionary of ground atoms to indexes
        """
        dic = {}
        for i, fact in enumerate(self.facts):
            dic[fact] = i
        return dic

    def build_head_unifier_dic(self):
        """
        build dictionary [(HEAD, FACT) -> THETA].

        Returns
        -------
        dic : {(.logic.Atom, .logic.Atom) -> List[(.logic.Var, .logic.Const)]}
            if the pair of atoms are unifiable, the dictional returns the unifier for them
        """
        dic = {}
        heads = set([c.head for c in self.clauses])
        for head in heads:
            for fi, fact in enumerate(self.facts):
                unify_flag, theta_list = unify([head, fact])
                if unify_flag:
                    dic[(head, fact)] = theta_list
        return dic

    def build_X_c(self, clause):
        """
        build index tensor for a given clause
        for ci in C, X_ci == X[i]
        X_c[i] is a list of indexes of facts that are needed to entail facts[i] using given clause.

        Inputs
        ------
        clasue : .logic.Clause
            input clause

        Returns
        -------
        X_c : torch.tensor((|facts|, max_body_len))
            index tensor for clause c
        """
        X_c = torch.zeros(
            (len(self.facts), self.max_body_len), dtype=torch.long).to(device)

        for fi, fact in enumerate(self.facts):
            if (clause.head, fact) in self.head_unifier_dic:
                theta_list = self.head_unifier_dic[(clause.head, fact)]
                clause_ = subs_list(clause, theta_list)
                need_facts = clause_.body
                if len(clause_.body) == 0:
                    need_indexes = [1]
                else:
                    need_indexes = [self.get_fact_index(
                        nf) for nf in need_facts]
                need_indexes_tensor = self.index_list_to_tensor(
                    need_indexes)
                X_c[fi, :] = need_indexes_tensor
        return X_c

    def index_list_to_tensor(self, index_list):
        """
        convert list of indexes to torch tensor 
        filling the gap by \top (See Eq.4 in the paper)

        Inputs
        ------
        index_list : List[int]
            list of indexes of ground atoms in the body

        Returns
        -------
        body_tensor : torch.tensor((max_body_len, ))
            index tensor for a clause
            indexes of ground atoms in the body after the unification of its head
        """
        diff = self.max_body_len - len(index_list)
        if diff > 0:
            return torch.tensor(index_list + [1 for i in range(diff)], dtype=torch.int32).to(device)
        else:
            return torch.tensor(index_list, dtype=torch.int32).to(device)

    def get_fact_index(self, fact):
        """
        convert fact to index in the ordered set of all facts

        Inputs
        ------
        fact : .logic.Atom
            ground atom

        Returns
        -------
        index : int
            index of the input ground atom
        """
        try:
            index = self.fact_index_dic[fact]
        except KeyError:
            index = 0
        return index
