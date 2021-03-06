import pickle
import itertools
from problem import ILPProblem
from language import Language
from logic import Const, FuncSymbol, Predicate, Atom, FuncTerm, Clause, Var
import random as random
#from numpy.random import *
import sys
sys.path.append('../')
sys.path.append('../../src/')


def list_to_term(ls, f):
    if len(ls) == 0:
        return Const('*')
    elif len(ls) == 1:
        return FuncTerm(f, [Const(str(ls[0])), Const('*')])
    else:
        return FuncTerm(f, [Const(str(ls[0])), list_to_term(ls[1:], f)])


def random_choices(ls, k):
    if k == 0:
        return []
    else:
        return [random.choice(ls) for i in range(k)]


def get_sublist(ls):
    if len(ls) == 1:
        return [ls] + [[]]
    else:
        return [ls] + get_sublist(ls[1:])


class AppendProblem(ILPProblem):
    def __init__(self, n=50, noise_rate=0.0):
        self.name = "append"
        self.pos_examples = []
        self.neg_examples = []
        self.backgrounds = []
        self.init_clauses = []
        p_ = Predicate('.', 1)
        false = Atom(p_, [Const('__F__')])
        true = Atom(p_, [Const('__T__')])
        self.facts = [false, true]
        self.lang = None
        self.noise_rate = noise_rate
        self.n = n
        self.max_len = 3
        self.symbols = list('abc')

    def get_pos_examples(self):
        i = 0
        while len(self.pos_examples) < self.n:
            n1 = random.randint(1, int(self.max_len))
            n2 = random.randint(1, int(self.max_len))
            _ls1 = random_choices(self.symbols, k=n1)
            ls2 = random_choices(self.symbols, k=n2)
            ls1_list = get_sublist(_ls1)
            for ls1 in ls1_list:
                ls3 = ls1 + ls2
                term1 = list_to_term(ls1, self.funcs[0])
                term2 = list_to_term(ls2, self.funcs[0])
                term3 = list_to_term(ls3, self.funcs[0])
                atom = Atom(self.preds[0], [term1, term2, term3])
                if not atom in self.pos_examples:
                    self.pos_examples.append(atom)
                    i += 1

                ls3 = ls2 + ls1
                term3 = list_to_term(ls3, self.funcs[0])
                atom = Atom(self.preds[0], [term2, term1, term3])
                if not atom in self.pos_examples:
                    self.pos_examples.append(atom)
                    i += 1

    def get_neg_examples(self):
        i = 0
        while i < self.n:
            n1 = random.randint(0, int(self.max_len))
            n2 = random.randint(0, int(self.max_len))
            n3 = random.randint(int(self.max_len/2), self.max_len)
            ls1 = random_choices(self.symbols, k=n1)
            ls2 = random_choices(self.symbols, k=n2)
            ls3 = random_choices(self.symbols, k=n3)
            term1 = list_to_term(ls1, self.funcs[0])
            term2 = list_to_term(ls2, self.funcs[0])
            term3 = list_to_term(ls3, self.funcs[0])

            if ls1 + ls2 != ls3:
                atom = Atom(self.preds[0], [term1, term2, term3])
                if not atom in self.neg_examples:
                    self.neg_examples.append(atom)
                    i += 1

    def get_backgrounds(self):
        # pass
        atom = Atom(self.preds[0], [Const('*'), Const('*'), Const('*')])
        self.backgrounds.append(atom)

    def get_clauses(self):
        clause1 = Clause(
            Atom(self.preds[0], [Var('X'), Var('Y'), Var('Z')]), [])
        self.clauses = [clause1]

    def get_facts(self):
        pass

    def get_templates(self):
        self.templates = [RuleTemplate(body_num=1, const_num=0),
                          RuleTemplate(body_num=0, const_num=1)]

    def get_language(self):
        self.preds = [Predicate('append', 3)]
        self.funcs = [FuncSymbol('f', 2)]
        self.consts = [Const(x) for x in self.symbols]
        self.lang = Language(preds=self.preds, funcs=self.funcs,
                             consts=self.consts, subs_consts=[Const('*')])


if __name__ == '__main__':
    problem = AppendProblem(n=50)
    problem.compile()
    problem.save_problem()
