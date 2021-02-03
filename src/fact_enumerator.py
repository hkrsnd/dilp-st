from logic import Predicate, Atom, Const
from logic_ops import unify, subs_list


class FactEnumerator():
    """
    fact enumerator for given ilp problem
    """

    @classmethod
    def enumerate_facts(self, Q, C, infer_step):
        """
        enumerate a set of necessary and sufficient ground atoms

        Inputs
        ------
        Q : (Set[.logic.Atom], Set[.logic.Atom], Set[.logic.Atom], Set[.logic.Atom])
            inductive logic programming problem (P, N, B, L)
        C : List[.logic.Clause]
            set of clauses from the clause-generation step
        infer_step : int
            number of steps of forward-chaining inference

        Returns
        -------
        G : List[.logic.Atom]
            set of enumerated ground atoms
        """
        pos_examples = Q.pos
        neg_examples = Q.neg
        backgrounds = Q.bk

        G = set(pos_examples + neg_examples + backgrounds)
        G_next = G
        G_past = G
        head_unifier_dic = {}

        for i in range(infer_step):
            S = set()
            for clause in C:
                for fi, fact in enumerate(G_next):
                    if (clause.head, fact) in head_unifier_dic:
                        unify_flag, theta_list = unify([fact, clause.head])
                    else:
                        unify_flag, theta_list = unify([fact, clause.head])
                        head_unifier_dic[(clause.head, fact)
                                         ] = unify_flag, theta_list
                    if unify_flag:
                        clause_ = subs_list(clause, theta_list)
                        B_i_list = clause_.body
                        S = S.union(set(B_i_list))
            G = G.union(S)
            G_past = G_past.union(G_next)
            G_next = S.difference(G_past)

        # SPECIAL
        p_ = Predicate('.', 1)
        false = Atom(p_, [Const('__F__')])
        true = Atom(p_, [Const('__T__')])

        G_given = set([false, true] + pos_examples +
                      neg_examples + backgrounds)
        G_inter = G.intersection(G_given)
        G_add = G.difference(G_inter)
        return [false, true] + backgrounds + pos_examples + neg_examples + list(G_add)
