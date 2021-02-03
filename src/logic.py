import itertools


def flatten(x): return [z for y in x for z in (
    flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class Term:
    """
    terms: variables, constants, or terms with function symbols
    """

    def __init__(self):
        self.name = 'void'

    def __eq__(self):
        return self == other


class Const(Term):
    """
    constants in logic

    Parameters
    ----------
    name : str
        name of the constant
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self, level=0):
        # ret = "\t"*level+repr(self.name)+"\n"
        ret = self.name
        return ret

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == Const:
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return self.name

    def head(self):
        return self

    def subs(self, target_var, const):
        return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for constant ' + str(self)
        return self

    def all_vars(self):
        return []

    def all_consts(self):
        return [self]

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1


class Var(Term):
    """
    variables in logic

    Parameters
    ----------
    name : str
        name of the variable
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self, level=0):
        # ret = "\t"*level+repr(self.name)+"\n"
        ret = self.name
        return ret

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == Var and self.name == other.name:
            return True
        else:
            return False

    def __hash__(self):
        return self.name

    def head(self):
        return self

    def subs(self, target_var, const):
        if self.name == target_var.name:
            return const
        else:
            return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for variable ' + str(self)
        return self

    def all_vars(self):
        return [self]

    def all_consts(self):
        return []

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1


class FuncSymbol():
    """
    function symbols in logic

    Parameters
    ----------
    name : str
        name of the function
    """

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name and self.arity == other.arity


class FuncTerm(Term):
    """
    term with a function symbol f(t_1, ..., t_n)

    Parameters
    ----------
    func_symbol : str
        name of the function
    args : List[Term]
        list of terms for the function
    """

    def __init__(self, func_symbol, args):
        assert func_symbol.arity == len(
            args), 'Invalid arguments for function symbol ' + func_symbol.name
        self.func_symbol = func_symbol
        self.args = args

    def __str__(self):
        s = self.func_symbol.name + '('
        for arg in self.args:
            s += arg.__str__() + ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self, level=0):
        return self.__str__()

    def __eq__(self, other):

        if type(other) == FuncTerm:
            if self.func_symbol != other.func_symbol:
                return False
            for i in range(len(self.args)):
                if not self.args[i] == other.args[i]:
                    return False
            return True
        else:
            return False

    def head(self):
        return self.func_symbol

    def pre_order(self, i):
        if i == 0:
            return self.func_symbol
        else:
            return self.pre_order(i-1)

    def get_ith_symbol(self, i):
        return self.to_list()[i]

    def get_ith_term(self, i):
        index = [0]
        result = [Term()]

        def _loop(x, i):
            nonlocal index, result
            if i == index[0]:
                result[0] = x
            else:
                if type(x) == FuncTerm:
                    for term in x.args:
                        index[0] += 1
                        _loop(term, i)
        _loop(self, i)
        return result[0]

    def to_list(self):
        ls = []

        def _to_list(x):
            nonlocal ls
            if type(x) == FuncTerm:
                ls.append(x.func_symbol)
                for term in x.args:
                    _to_list(term)
            else:
                # const or var
                ls.append(x)
        _to_list(self)
        return ls

    def subs(self, target_var, const):
        self.args = [arg.subs(target_var, const) for arg in self.args]
        return self

    def all_vars(self):
        var_list = []
        for arg in self.args:
            var_list += arg.all_vars()
        return var_list

    def all_consts(self):
        const_list = []
        for arg in self.args:
            const_list += arg.all_consts()
        return const_list

    def all_funcs(self):
        func_list = []
        for arg in self.args:
            func_list += arg.all_funcs()
        return [self.func_symbol] + func_list

    def max_depth(self):
        arg_depth = max([arg.max_depth() for arg in self.args])
        return arg_depth+1

    def min_depth(self):
        arg_depth = min([arg.min_depth() for arg in self.args])
        return arg_depth+1

    def size(self):
        size = 1
        for arg in self.args:
            size += arg.size()
        return size


class Predicate():
    """
    predicate symbol

    Parameters
    ----------
    name : str
        name of the predicate
    arity : int
        arity of the predicate (number of args)
    """

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __str__(self):
        return self.name

    def __expr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == Predicate:
            return self.name == other.name
        else:
            return False


class Atom():
    """
    atoms in logic p(t1, ..., tn)

    Parameters
    ----------
    pred : Predicate
        predicate of the atom
    terms : List[Term]
        terms for the predicate
    """

    def __init__(self, pred, terms):
        assert pred.arity == len(
            terms), 'Invalid arguments for predicate symbol ' + pred.name
        self.pred = pred
        self.terms = terms
        self.neg_state = False

    def __eq__(self, other):
        if self.pred == other.pred:
            for i in range(len(self.terms)):
                if not self.terms[i] == other.terms[i]:
                    return False
            return True
        else:
            return False

    def __str__(self):
        s = self.pred.name + '('
        for arg in self.terms:
            s += arg.__str__() + ','
        s = s[0:-1]
        s += ')'
        return s

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def subs(self, target_var, const):
        self.terms = [term.subs(target_var, const) for term in self.terms]

    def neg(self):
        self.neg_state = not self.neg_state

    def all_vars(self):
        var_list = []
        for term in self.terms:
            # var_list.append(term.all_vars())
            var_list += term.all_vars()
        return var_list

    def all_consts(self):
        const_list = []
        for term in self.terms:
            const_list += term.all_consts()
        return const_list

    def all_funcs(self):
        func_list = []
        for term in self.terms:
            func_list += term.all_funcs()
        return func_list

    def max_depth(self):
        return max([term.max_depth() for term in self.terms])

    def min_depth(self):
        return min([term.min_depth() for term in self.terms])

    def size(self):
        size = 0
        for term in self.terms:
            size += term.size()
        return size


class EmptyAtom(Atom):
    """
    empty atom
    """

    def __init__(self):
        self.neg_state = False

    def __eq__(self, other):
        if type(other) == EmptyAtom:
            return True
        else:
            return False

    def __str__(self):
        return ''

    def __repr__(self):
        return self.__str__()

    def subs(self, target_var, const):
        return self

    def neg(self):
        return self

    def all_vars(self):
        return []

    def all_consts(self):
        return []

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def max_depth(self):
        return 0

    def size(self):
        return 0


class Clause():
    """
    clause A :- B1,...,Bn.

    Parameters
    ----------
    head : Atom
        head atom A
    body : List[Atom]
        body atoms [B1, ..., Bn]
    """

    def __init__(self, head, body):
        self.head = head
        self.body = body
        # ['X' + str(i) for i in range(100)]
        self.var_names = ['X', 'Y', 'Z', 'V', 'W', 'A', 'B', 'C']
        self.var_list = [Var(name) for name in self.var_names]
        self.dummy_var_list = [Var(name+'__') for name in self.var_names]
        # self.rename_vars()

    def __str__(self):
        head_str = self.head.__str__()
        body_str = ""
        for bi in self.body:
            body_str += bi.__str__()
            body_str += ','
        body_str = body_str[0:-1]
        body_str += '.'
        return head_str + ':-' + body_str

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()
        # return self.head == other.head and self.body == other.body

    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        return self.__str__() > other.__str__()

    def rename_vars(self):
        """
        rename the var names to evaluate the equality.
        """
        clause_var_list = self.all_vars()
        for v in clause_var_list:
            if v in self.var_list:
                # replace to dummy to avoid conflicts
                # AVOID: p(x1,x2) :- p(X,Y) => p(X,x2) :- p(X,Y)
                dummy_index = self.var_list.index(v)
                dummy_v = self.dummy_var_list[dummy_index]
                self.subs(v, dummy_v)

        clause_var_list = self.all_vars()
        for i, v in enumerate(clause_var_list):
            self.subs(v, self.var_list[i])

    def is_tautology(self):
        if len(self.body) == 1 and self.body[0] == self.head:
            return True
        else:
            return False

    def is_duplicate(self):
        if len(self.body) >= 2:
            es = self.body
            return es == [es[0]] * len(es) if es else False
        return False

    def resolution(self, other):
        """
        A resolution function between two definite clauses.
        """
        # remove if p(x) and neg p(x)
        pos_atoms = [self.head, other.head]
        neg_atoms = self.body + other.body
        atoms_to_remove = []
        for pos_atom in pos_atoms:
            if pos_atom in neg_atoms:
                atoms_to_remove.append(pos_atom)
        resulted_clause = Clause()
        # compress same literals
        return resulted_clause

    def subs(self, target_var, const):
        if type(self.head) == Atom:
            self.head.subs(target_var, const)
        for bi in self.body:
            bi.subs(target_var, const)

    def all_vars(self):
        var_list = []
        var_list += self.head.all_vars()
        for bi in self.body:
            var_list += bi.all_vars()
        var_list = flatten(var_list)
        # remove duplication
        result = []
        for v in var_list:
            if not v in result:
                result.append(v)
        return result

    def all_consts(self):
        const_list = []
        const_list += self.head.all_consts()
        for bi in self.body:
            const_list += bi.all_consts()
        const_list = flatten(const_list)
        return const_list

    def all_funcs(self):
        func_list = []
        func_list += self.head.all_funcs()
        for bi in self.body:
            func_list += bi.all_funcs()
        func_list = flatten(func_list)
        return func_list

    def max_depth(self):
        depth_list = [self.head.max_depth()]
        for b in self.body:
            depth_list.append(b.max_depth())
        return max(depth_list)

    def min_depth(self):
        depth_list = [self.head.min_depth()]
        for b in self.body:
            depth_list.append(b.min_depth())
        return min(depth_list)

    def size(self):
        size = self.head.size()
        for bi in self.body:
            size += bi.size()
        return size


class EmptyClause(Clause):
    """
    empty clause □
    used in the SLD tree
    """

    def __init__(self):
        self.head = []
        self.body = []

    def __str__(self):
        return '□'

    def __repr__(self):
        return '□'
