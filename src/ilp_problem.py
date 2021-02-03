class ILPProblem():
    """
    inductive logic programming problem

    Parameters
    ----------
    pos : List[.logic.Atom]
    neg : List[.logic.Atom]
    bk : List[.logic.Atom]
    lang : List[.language.Language]
    name : str
        name of the problem
    """
    def __init__(self, pos, neg, bk, lang, name=""):
        self.pos = pos
        self.neg = neg
        self.bk = bk
        self.lang = lang
        self.name = name

    def print(self):
        """
        print summary of the problem
        """
        print('======= POSITIVE EXAMPLES =======')
        print(self.pos)
        print('======= NEGATIVE EXAMPLES =======')
        print(self.neg)
        print('======= BACKGROUND KNOWLEDGE  =======')
        print(self.bk)
