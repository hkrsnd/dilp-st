from logic import Var


class Language():
    """
    Language of logic programs

    Parameters
    ----------
    preds : List[.logic.Predicate]
        set of predicate symbols
    funcs : List[.logic.FunctionSymbol]
        set of function symbols
    consts : List[.logic.Const]
        set of constants
    subs_consts : List[.logic.Const]
        set of constants that can be substituted in the refinement step
    """

    def __init__(self, preds, funcs, consts, subs_consts=[]):
        self.preds = preds
        self.funcs = funcs
        self.consts = consts
        self.subs_consts = subs_consts
        self.var_gen = VariableGenerator()


class VariableGenerator():
    """
    generator of variables

    Parameters
    __________
    base_name : str
        base name of variables
    """

    def __init__(self, base_name='x'):
        self.counter = 0
        self.base_name = base_name

    def generate(self):
        """
        generate variable with new name

        Returns
        -------
        generated_var : .logic.Var
            generated variable
        """
        generated_var = Var(self.base_name + str(self.counter))
        self.counter += 1
        return generated_var
