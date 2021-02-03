import random
import pickle
import itertools
from numpy.random import *
from problem import ILPProblem
from language import Language
from logic import Const, FuncSymbol, Predicate, Atom, FuncTerm, Clause, Var
import sys
sys.path.append('../')
sys.path.append('../../src/')


def get_products(n):
    ls = list(range(n))
    return list(itertools.product(ls, ls, ls))


def get_num(n):
    return randint(n)


def int_to_term(n):
    s = FuncSymbol('s', 1)
    zero = Const('0')

    num = zero
    for i in range(n):
        num = FuncTerm(s, [num])
    return num


class PlusProblem(ILPProblem):
    def __init__(self, n=50, noise_rate=0.0):
        self.name = "plus"
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
        self.max_n = 7
        self.symbols = list('0')

    def get_pos_examples(self):
        while(len(self.pos_examples) < self.n):
            i = random.randint(0, self.max_n)
            j = random.randint(0, self.max_n)
            sum_ = i+j
            atom = Atom(self.preds[0], [int_to_term(
                i), int_to_term(j), int_to_term(sum_)])
            if not atom in self.pos_examples:
                if not (i == 0 and j == 0):
                    self.pos_examples.append(atom)

    def get_neg_examples(self):
        while(len(self.neg_examples) < self.n):
            i = random.randint(0, self.max_n)
            j = random.randint(0, self.max_n)
            sum_ = random.randint(0, self.max_n*2)
            atom = Atom(self.preds[0], [int_to_term(
                i), int_to_term(j), int_to_term(sum_)])
            if not atom in self.neg_examples and i+j != sum_:
                self.neg_examples.append(atom)

    def __get_pos_examples(self):
        for i in range(self.max_n):
            for j in range(self.max_n):
                # if i == 0 and j == 0:
                #    continue
                sum_ = i + j
                atom = Atom(self.preds[0], [int_to_term(
                    i), int_to_term(j), int_to_term(sum_)])
                if not atom in self.pos_examples:
                    self.pos_examples.append(atom)
                atom = Atom(self.preds[0], [int_to_term(
                    j), int_to_term(i), int_to_term(sum_)])
                if not atom in self.pos_examples:
                    self.pos_examples.append(atom)

    def __get_neg_examples(self):
        for i in range(self.max_n):
            for j in range(self.max_n):
                if (i != 0 or j != 0):
                    sum_ = i + j
                    if sum_+1 <= self.max_n + self.max_n:
                        self.neg_examples.append(
                            Atom(self.preds[0], [int_to_term(i), int_to_term(j), int_to_term(sum_+1)]))
                    if sum_-1 >= 0:
                        self.neg_examples.append(
                            Atom(self.preds[0], [int_to_term(i), int_to_term(j), int_to_term(sum_-1)]))

    def get_backgrounds(self):
        # pass
        zero = Const('0')
        self.backgrounds.append(Atom(self.preds[0], [zero, zero, zero]))

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
        self.preds = [Predicate('plus', 3)]
        self.funcs = [FuncSymbol('s', 1)]
        self.consts = [Const(x) for x in self.symbols]
        self.lang = Language(preds=self.preds, funcs=self.funcs,
                             consts=self.consts,  subs_consts=[Const('0')])


if __name__ == '__main__':
    problem = PlusProblem(n=50)
    problem.compile()
    problem.save_problem()
