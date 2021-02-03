import random
import itertools
from problem import ILPProblem
from language import Language
from logic import Const, FuncSymbol, Predicate, Atom, FuncTerm, Clause, Var
import random as random
import sys
sys.path.append('../')
sys.path.append('../../src/')


def flatten(x): return [z for y in x for z in (
    flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


def list_to_term(ls, f):
    if len(ls) == 1:
        return Const(str(ls[0]))
    else:
        return FuncTerm(f, [Const(str(ls[0])), list_to_term(ls[1:], f)])


def list_to_terms(ls, f):
    if len(ls) >= 3:
        return [list_to_term(ls, f)] + list_to_terms(ls[1:], f)
    elif len(ls) == 2:
        return [list_to_term]


def list_to_examples(ls, pred, f):
    atoms = []
    for x in ls:
        list_term = list_to_term(ls, f)
        atom = Atom(pred, [Const(x), list_term])
        atoms.append(atom)
    return atoms


def get_sublist(ls):
    if len(ls) == 2:
        return [ls]
    else:
        return [ls] + get_sublist(ls[1:])


def random_choices(ls, k):
    return [random.choice(ls) for i in range(k)]


class Node():
    def __init__(self, label, children):
        self.label = label
        self.children = children

    def __str__(self):
        if len(self.children) > 0:
            s = self.label + '('
            for arg in self.children:
                s += arg.__str__() + ','
            s = s[0:-1]
            s += ')'
        else:
            s = self.label
        return s

    def __repr__(self, level=0):
        return self.__str__()

    def size(self):
        size = 1
        for c in self.children:
            size += c.size()
        return size

    def is_subtree(self, other):
        print(self, other, self.__str__() in other.__str__())
        return self.__str__() in other.__str__()


def random_tree(n=6, labels=list('abcd')):
    root = Node(random.choice(labels), [])
    while root.size() <= n-1:
        add_child_random(root, labels)
    return root


def add_child_random(node, labels):
    if len(node.children) == 0:
        node.children.append(Node(random.choice(labels), []))
    if len(node.children) == 1:
        node.children.append(Node(random.choice(labels), []))
    else:
        c = random.choice(node.children)
        add_child_random(c, labels)


def all_subtrees(node):
    if len(node.children) == 0:
        return []
    else:
        return [node] + flatten([all_subtrees(c) for c in node.children])


def tree_to_term(tree, f, phi):
    if len(tree.children) == 2:
        return FuncTerm(f, [tree_to_term(c, f, phi) for c in tree.children])
    elif len(tree.children) == 1:
        return FuncTerm(f, [tree_to_term(tree.children[0], f, phi), Const(phi)])
    else:
        return Const(tree.label)


def to_atom(t1, t2, pred):
    return Atom(pred, [t1, t2])


class SubtreeProblem(ILPProblem):
    def __init__(self, n=30, noise_rate=0.0):
        self.name = "subtree"
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
        self.max_size = 12
        self.symbols = list('abc')
        self.phi = '*'

    def __get_pos_examples(self):
        while len(self.pos_examples) < self.n:
            tree = random_tree(n=self.max_size, labels=self.symbols)
            subtrees = all_subtrees(tree)
            tree_term = tree_to_term(tree, self.funcs[0], self.phi)
            for st in subtrees:
                st_term = tree_to_term(st, self.funcs[0], self.phi)
                if st_term.__str__() in tree_term.__str__():
                    e = to_atom(st_term, tree_term, self.preds[0])
                    self.pos_examples.append(e)

    def get_pos_examples(self):
        while len(self.pos_examples) < self.n:
            n1 = random.randint(2, self.max_size)
            n2 = random.randint(n1, self.max_size)
            t1 = random_tree(n=n1, labels=self.symbols)
            t2 = random_tree(n=n2, labels=self.symbols)
            t1_term = tree_to_term(t1, self.funcs[0], self.phi)
            subtrees = all_subtrees(t1)
            t2 = random.choice(subtrees)
            t2_term = tree_to_term(t2, self.funcs[0], self.phi)
            e = to_atom(t2_term, t1_term, self.preds[0])
            if t2.__str__() in t1.__str__():
                self.pos_examples.append(e)

    def get_neg_examples(self):
        while len(self.neg_examples) < self.n:
            n1 = random.randint(1, self.max_size)
            n2 = random.randint(n1, self.max_size)
            t1 = random_tree(n=n1, labels=self.symbols)
            t2 = random_tree(n=n2, labels=self.symbols)
            t1_term = tree_to_term(t1, self.funcs[0], self.phi)
            t2_term = tree_to_term(t2, self.funcs[0], self.phi)
            e = to_atom(t1_term, t2_term, self.preds[0])
            if not t1.__str__() in t2.__str__():
                self.neg_examples.append(e)

    def get_backgrounds(self):
        # for s in self.symbols:
        #    atom = Atom(self.preds[0], [Const(s), Const(s)])
        #    self.backgrounds.append(atom)
        for s in self.symbols:
            self.backgrounds.append(Atom(self.preds[0], [Const(s), Const(s)]))

    def get_clauses(self):
        clause1 = Clause(Atom(self.preds[0], [Var('X'), Var('Y')]), [])
        self.clauses = [clause1]

    def get_templates(self):
        self.templates = [RuleTemplate(body_num=1, const_num=0),
                          RuleTemplate(body_num=0, const_num=0),
                          RuleTemplate(body_num=0, const_num=0)]

    def get_language(self):
        self.preds = [Predicate('subtree', 2)]
        self.funcs = [FuncSymbol('f', 2)]
        self.consts = [Const(x) for x in self.symbols]
        self.lang = Language(preds=self.preds, funcs=self.funcs,
                             consts=self.consts)


if __name__ == '__main__':
    #print(Node('a', []))
    problem = SubtreeProblem(n=50, noise_rate=0.0)
    problem.compile()
    problem.save_problem()
