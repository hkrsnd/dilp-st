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


class MemberProblem(ILPProblem):
    def __init__(self, n=30, noise_rate=0.0):
        self.name = "member"
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
        self.max_len = 4
        self.symbols = list('abc')

    def get_pos_examples(self):
        i = 0
        while len(self.pos_examples) < self.n:
            n = random.randint(2, self.max_len)
            x = random.choice(self.symbols)
            ls = random_choices(self.symbols, k=n)
            if x in ls:
                term1 = Const(x)
                term2 = list_to_term(ls, self.funcs[0])
                atom = Atom(self.preds[0], [term1, term2])
                self.pos_examples.append(atom)

    def get_neg_examples(self):
        i = 0
        while i < self.n:
            n = random.randint(1, self.max_len)
            # 長さnで満たすもの出すまで繰り返し
            flag = True
            while flag:
                x = random.choice(self.symbols)
                ls = random_choices(self.symbols, n)
                if not x in ls:
                    atom = Atom(self.preds[0], [
                                Const(x), list_to_term(ls, self.funcs[0])])
                    self.neg_examples.append(atom)
                    i += 1
                    flag = False

    def get_backgrounds(self):
        # pass
        #self.backgrounds.append(Atom(self.preds[0], [Const('*'), Const('*')]))
        for s in self.symbols:
            atom = Atom(self.preds[0], [
                        Const(s), list_to_term([s], self.funcs[0])])
            self.backgrounds.append(atom)
        #     self.backgrounds.append(atom)
        # for s in self.symbols:
        #    self.backgrounds.append(Atom(self.preds[0], [Const(s), Const(s)]))

    def get_clauses(self):
        clause1 = Clause(Atom(self.preds[0], [Var('X'), Var('Y')]), [])
        self.clauses = [clause1]

    def get_facts(self):
        terms = []
        for i in range(1, self.max_len+1):
            i_len_list = list(itertools.product(self.symbols, repeat=i))
            for l in i_len_list:
                term = list_to_term(l, self.funcs[0])
                terms.append(term)
        # generate facts
        args1 = [term for term in terms if term.max_depth() <= 0]
        args2 = [term for term in terms]
        for pair in list(itertools.product(args1, args2)):
            self.facts.append(Atom(self.preds[0], list(pair)))

    def get_templates(self):
        self.templates = [RuleTemplate(body_num=1, const_num=0),
                          RuleTemplate(body_num=0, const_num=0)]

    def get_language(self):
        self.preds = [Predicate('member', 2)]
        self.funcs = [FuncSymbol('f', 2)]
        self.consts = [Const(x) for x in self.symbols]
        self.lang = Language(preds=self.preds, funcs=self.funcs,
                             consts=self.consts)


if __name__ == '__main__':
    problem = MemberProblem(n=50)
    problem.compile()
    problem.save_problem()
