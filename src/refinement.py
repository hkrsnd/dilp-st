import itertools
from logic import Atom, Clause, FuncTerm, Var
from logic_ops import subs


class RefinementGenerator():
    """
    refinement operations for clause generation

    Parameters
    ----------
    lang : .language.Language
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, lang, max_depth=1, max_body_len=1):
        self.lang = lang
        self.max_depth = max_depth
        self.max_body_len = max_body_len

    def refinement_clauses(self, C):
        """
        apply refinement operations to each element in given set of clauses

        Inputs
        ------
        C : List[.logic.Clause]
            set of clauses

        Returns
        -------
        C_refined : List[.logic.Clause]
            refined clauses
        """
        C_refined = []
        for clause in C:
            C_refined.extend(self.refinement(clause))
        return list(set(C_refined))

    def refinement(self, clause):
        """
        refinement operator that consist of 4 types of refinement

        Inputs
        ______
        clause : .logic.Clause
            input clause to be refined

        Returns
        -------
        refined_clauses : List[.logic.Clause]
            refined clauses
        """
        refs = list(set(self.add_atom(clause) + self.apply_func(clause) +
                        self.subs_var(clause) + self.subs_const(clause)))
        result = []
        for ref in refs:
            if not '' in [str(arg) for arg in ref.head.terms]:
                result.append(ref)
        return result

    def add_atom(self, clause):
        """
        add p(x_1, ..., x_n) to the body        
        """
        # Check body length
        if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
            return []

        refined_clauses = []
        for p in self.lang.preds:
            var_candidates = clause.all_vars()
            # Select X_1, ..., X_n for new atom p(X_1, ..., X_n)
            # 1. Selection 2. Ordering
            for vs in itertools.permutations(var_candidates, p.arity):
                new_atom = Atom(p, vs)
                head = clause.head
                if new_atom != head and not (new_atom in clause.body):
                    new_body = clause.body + [new_atom]
                    new_clause = Clause(head, new_body)
                    refined_clauses.append(new_clause)
        return refined_clauses

    def apply_func(self, clause):
        """
        z/f(x_1, ..., x_n) for every variable in C and every n-ary function symbol f in the language        
        """
        refined_clauses = []
        if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
            return []

        funcs = clause.all_funcs()
        for z in clause.head.all_vars():
            # for z in clause.all_vars():
            for f in self.lang.funcs:
                # if len(funcs) >= 1 and not(f in funcs):
                #    continue

                new_vars = [self.lang.var_gen.generate()
                            for v in range(f.arity)]
                func_term = FuncTerm(f, new_vars)
                # TODO: check variable z's depth
                result = subs(clause, z, func_term)
                if result.max_depth() <= self.max_depth:
                    result.rename_vars()
                    refined_clauses.append(result)
        return refined_clauses

    def subs_var(self, clause):
        """
        z/x for every distinct variables x and z in C
        """
        refined_clauses = []
        # to HEAD
        all_vars = clause.head.all_vars()
        combs = itertools.combinations(all_vars, 2)
        for u, v in combs:
            result = subs(clause, u, v)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses

    def subs_const(self, clause):
        """
        z/a for every variable z in C and every constant a in the language
        """
        if (len(clause.body) >= self.max_body_len) or (clause.max_depth() >= 1):
            return []

        refined_clauses = []
        all_vars = clause.head.all_vars()
        consts = self.lang.subs_consts
        for v, c in itertools.product(all_vars, consts):
            result = subs(clause, v, c)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses
