class ILPProblem():
    def __init__(self, name, const_subs=False, noise_rate=0.0):
        self.name = ""
        self.pos_examples = []
        self.neg_examples = []
        self.backgrounds = []
        self.init_clauses = []
        self.facts = []
        self.lang = None
        self.const_subs = const_subs
        self.noise_rate = noise_rate

    def compile(self):
        self.get_language()
        self.get_clauses()
        self.get_pos_examples()
        self.get_neg_examples()
        self.get_backgrounds()

    def get_pos_examples(self):
        pass

    def get_neg_examples(self):
        pass

    def get_backgrounds(self):
        pass

    def get_clauses(self):
        pass

    def get_facts(self):
        pass

    def get_language(self):
        pass

    def save_problem(self):
        base_path = './' + self.name
        self.save_list(base_path+'.pos', self.pos_examples)
        self.save_list(base_path+'.neg', self.neg_examples)
        self.save_list(base_path+'.bk', self.backgrounds)
        self.save_list(base_path+'.cl', self.clauses)
        self.save_language(base_path+'.lang', self.lang)

    def save_list(self, path, ls):
        with open(path, 'w') as f:
            for x in ls:
                f.write(str(x) + '\n')

    def save_language(self, path, lang):
        # append:2, plus:3
        # f:2, s:1
        # a,b,0
        with open(path, 'w') as f:
            pred_str = ""
            func_str = ""
            const_str = ""
            subs_const_str = ""
            for pred in lang.preds:
                pred_str += str(pred)+':'+str(pred.arity)+','
            for func in lang.funcs:
                func_str += str(func)+':'+str(func.arity)+','
            for const in lang.consts:
                const_str += str(const) + ','
            for s_const in lang.subs_consts:
                subs_const_str += str(s_const) + ','
            f.write(pred_str[:-1] + '\n')
            f.write(func_str[:-1] + '\n')
            f.write(const_str[:-1] + '\n')
            f.write(subs_const_str[:-1] + '\n')
