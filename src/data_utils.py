from lark import Lark
from exp_parser import ExpTree
from language import Language
from logic import Predicate, FuncSymbol, Const


class DataUtils():
    '''
    Functions to read data from files.
    '''

    def __init__(self, name):
        self.name = name
        base_path = 'datasets/' + self.name + '/'
        self.lang = self.load_lang(base_path + self.name+'.lang')
        with open("src/exp.lark", encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open("src/exp.lark", encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    def load_atoms(self, path):
        '''
        Read lines and parse to Atom objects.
        '''
        atoms = []
        with open(path) as f:
            for line in f:
                tree = self.lp_atom.parse(line[:-1])
                atom = ExpTree(self.lang).transform(tree)
                atoms.append(atom)
        return atoms

    def load_lang(self, path):
        '''
        Read lines and parse to languages
        '''
        f = open(path)
        lines = f.readlines()
        f.close()
        preds = self.parse_preds(lines[0][:-1])
        funcs = self.parse_funcs(lines[1][:-1])
        consts = self.parse_consts(lines[2][:-1])
        subs_consts = self.parse_consts(lines[3][:-1])
        return Language(preds, funcs, consts,  subs_consts)

    def parse_preds(self, line):
        '''
        Parse string to predicates
        '''
        preds = []
        for pred_arity in line.split(','):
            pred, arity = pred_arity.split(':')
            preds.append(Predicate(pred, int(arity)))
        return preds

    def parse_funcs(self, line):
        '''
        Parse string to function symbols
        '''
        funcs = []
        for func_arity in line.split(','):
            func, arity = func_arity.split(':')
            funcs.append(FuncSymbol(func, int(arity)))
        return funcs

    def parse_consts(self, line):
        '''
        Parse string to function symbols
        '''
        return [Const(x) for x in line.split(',')]

    def load_clauses(self, path):
        '''
        Read lines and parse to Atom objects.
        '''
        clauses = []
        with open(path) as f:
            for line in f:
                tree = self.lp_clause.parse(line[:-1])
                clause = ExpTree(self.lang).transform(tree)
                clauses.append(clause)
        return clauses

    def load_data(self):
        '''
        Read files and return an ILP problem.
        '''
        base_path = 'datasets/' + self.name + '/'
        pos = self.load_atoms(base_path + self.name+'.pos')
        neg = self.load_atoms(base_path + self.name+'.neg')
        bk = self.load_atoms(base_path + self.name+'.bk')
        clauses = self.load_clauses(base_path + self.name+'.cl')
        return pos, neg, bk, clauses, self.lang
