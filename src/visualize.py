import os
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')


class Visualize():
    """
    Visualization functions to plot results of computations

    Parameters
    ----------
    dilp : .ilp_problem.ILPProblem
        ilp problem
    name : str
        name of the ilp problem
    facts : List[.logic.Atom]
        set of ground atoms
    """

    def __init__(self, dilp, name, facts):
        self.dilp = dilp
        self.name = name
        self.facts = facts
        self.bk_indexes = [facts.index(
            p) for p in self.dilp.bk]
        self.pos_indexes = [facts.index(
            p) for p in self.dilp.pos]
        self.neg_indexes = [facts.index(
            p) for p in self.dilp.neg]
        self.indexes = np.array([0, 1] + self.pos_indexes + self.neg_indexes)

    def get_example_valuations(self, valuation):
        ex_valuation = [valuation[0], valuation[1]]
        for i in self.bk_indexes:
            ex_valuation.apend(valuation[i])
        for i in self.pos_indexes:
            ex_valuation.append(valuation[i])
        for i in self.neg_indexes:
            ex_valuation.append(valuation[i])

        return np.array(ex_valuation)

    def plot_weights(self, Ws, epoch=0):
        sns.set()
        sns.set_style('white')
        fig = plt.figure()
        plt.ylim([0, 1])
        for i, W in enumerate(Ws):
            W_ = W.detach().cpu().numpy()
            plt.bar(range(len(W_)), W_, alpha=0.8, label='C' + str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.savefig('imgs/weights/' + self.name + '/' + 'W_' +
                    str(epoch) + '.png',  bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_valuation_memory(self, at_list, epoch=0):
        infer_mat = np.array([at.detach().cpu().numpy() for at in at_list])
        #plot_mat = [self.gfet_example_valuations(row) for row in infer_mat]
        plot_fact_num = 2 + len(self.bk_indexes) + \
            len(self.pos_indexes) + len(self.neg_indexes)
        plot_mat = [row[:plot_fact_num] for row in infer_mat]
        last_row = plot_mat[-1]
        ######result_row = [1.0 if x >= 0.5 else 0.0 for x in last_row]
        base_row = np.array([0.0, 1.0] + [1.0 for i in range(len(self.bk_indexes))] + [
                            1.0 for i in range(len(self.pos_indexes))] + [0.0 for i in range(len(self.neg_indexes))])
        diff_row = np.abs(last_row - base_row)

        plot_mat = list(plot_mat)
        ######plot_mat.append(np.array(result_row, dtype=np.float32))
        plot_mat.append(np.array(diff_row, dtype=np.float32))

        plot_mat = np.array(plot_mat)

        #fig, ax = plt.subplots()
        plt.rcParams["axes.grid"] = False

        fig = plt.figure(figsize=(20, 7))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(plot_mat, interpolation='none')
        plt.colorbar()

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(plot_mat[0])))
        ax.set_yticks(np.arange(len(at_list) + 1))
        y_ticks = ['v_' + str(i)
            for i in range(len(at_list))] + ['diff']
        ######           for i in range(len(at_list))] + ['class', 'diff']
        ax.set_yticklabels(y_ticks)
        # ... and label them with the respective list entries
        #ax.set_xticklabels([c.__str__() for c in self.dilp.clauses])
        spec_strs = [self.facts[0].__str__(),
                     self.facts[1].__str__()]
        bk_strs = []
        for i in self.bk_indexes:
            bk_strs.append(self.facts[i].__str__() + '*')
        pos_strs = []
        for i in self.pos_indexes:
            pos_strs.append(self.facts[i].__str__() + '+')
        neg_strs = []
        for i in self.neg_indexes:
            neg_strs.append(self.facts[i].__str__() + '-')
        #pos_strs = [e.__str__() + '+' for e in self.dilp.pos]
        #neg_strs = [e.__str__() + '-' for e in self.dilp.neg]
        facts_strs = spec_strs + bk_strs + pos_strs + neg_strs
        ax.set_xticklabels(facts_strs)

        #ax.set_yticklabels(list(range(1, self.dilp.infer_steps+1)))

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        '''
        for i in range(len(at_list)):
            for j in range(len(at_list[0])):
                text = ax.text(j, i, infer_mat[i, j],
                               ha="center", va="center", color="w")
        '''

        #ax.set_title("Inference Steps")
        # fig.tight_layout()
        plt.show()
        #plt.savefig(self.name + '.png')
        plt.close()
        #os.makedirs('imgs/' + self.name, exist_ok=True)
        #plt.savefig('imgs/' + self.name + '/infer_' + str(epoch) + '.png')
