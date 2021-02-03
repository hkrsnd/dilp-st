import torch.nn.functional as F
import torch
import random
random.seed(a=7014)  # PAPER ID 7014


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def convert(bk, facts):
    """
    convert background knowledge to the valuation vector

    Inputs
    ------
    bk : List[.logic.Atom]
        background knowledge
    facts : List[.logic.Atom]
        enumerated ground atoms

    Returns
    -------
    v_0 : torch.Tensor((|facts|, ))
        initial valuation vector
    """
    return torch.tensor([1.0 if (fact in bk or str(fact) == '.(__T__)') else 0.0 for fact in facts], dtype=torch.float32).to(device)


class InferModule():
    """
    differentiable inference module

    Parameters
    ----------
    X : torch.Tensor((|clauses|, |facts|, max_body_len, ))
        index tensor
    m : int
        size of the logic program
    v_0 : torch.Tensor((|facts|, ))
        initial valuation vector
    infer_step : int
        number of steps in forward-chaining inference
    """

    def __init__(self, X, m, v_0, infer_step):
        self.X = X
        self.Ws = self.init_weights(m)
        self.max_body_len = len(X[0][0])
        self.v_0 = v_0
        self.infer_step = infer_step

    def init_weights(self, m):
        """
        initialize weights randomly

        Inputs
        ------
        m : int
            size of the logic program
        """
        return [torch.tensor([random.random() for i in range(len(self.X))]).to(
            device).detach().requires_grad_(True)
            for j in range(m)]

    def infer(self):
        """
        f_infer function
        compute v_0, v_1, ..., v_T and return v_T

        Returns
        -------
        v_T : torch.tensor((|facts|, ))
            valuation vector of the result of the forward-chaining inference
        """
        self.valuation_memory = [self.v_0]
        valuation = self.valuation_memory[0]

        n = len(self.Ws)

        Ws_softmaxed = [self.softmax(W) for W in self.Ws]

        for t in range(self.infer_step):
            F_c_tensor = torch.stack([self.F_c(ci, valuation)
                                      for ci in range(len(self.X))], 0)
            b_t_list = [torch.matmul(W, F_c_tensor) for W in Ws_softmaxed]
            b_t = self.softor(b_t_list)
            # print(b_t)
            valuation = self.amalgamate(valuation, b_t)
            self.valuation_memory.append(valuation)

        if (valuation > 1.0).any():
            valuation = valuation / torch.max(valuation)

        return valuation

    def F_c(self, ci, valuation):
        """
        c_i function
        forward-chaining inference using a clause

        Inputs
        ------
        ci : .logic.Clause
            i-th clause in the set of enumerated clauses
        valuation : torch.tensor((|facts|, ))
            current valuation vector v_t

        Returns
        -------
        v_{t+1} : torch.tensor((|facts|, ))
            valuation vector after 1-step forward-chaining inference
        """
        X_c = self.X[ci, :, :]
        gathered = self.gather(valuation, X_c)
        valuation_ = self.prod_body(gathered)
        return valuation_

    def softmax(self, x, beta=1.0):
        """
        softmax fuction for torch vectors
        """
        return F.softmax(x / beta, dim=0)

    def amalgamate(self, x, y):
        """
        amalgamate function for valuation vectors
        """
        return self.softor([x, y])

    def gather(self, x, y):
        """
        gather function for torch tensors
        """
        tensors = [torch.gather(x, 0, y[:, i]).unsqueeze(-1)
                   for i in range(self.max_body_len)]
        return torch.cat(tensors, -1).to(device)

    def softor(self, xs, gamma=1e-5):
        """
        softor function for valuation vectors
        """
        xs_tensor = torch.stack(xs, 1) * (1/gamma)
        return gamma*torch.logsumexp(xs_tensor, dim=1)

    def prod_body(self, gathered):
        """
        taking product along the body atoms

        Inputs
        ------
        gathered : torch.tensor(())

        Returns
        -------
        result : torch.tensor(())
        """
        result = torch.ones(self.valuation_memory[0].shape).to(device)
        result[0] = 0.0  # False = 0.0
        for i in range(self.max_body_len):
            result = result * gathered[:, i]
        return result


class InferModulePair(InferModule):
    """
    differentiable inference module with logic programs that consist of only pairs of clauses (d-ILP style)
    """

    def init_weights(self, m):
        """
        initialize weights randomly

        Inputs
        ------
        m : int
            size of the logic program
        """
        return [torch.tensor([[random.random() for i in range(len(self.X))]
                              for j in range(len(self.X))]).to(device).detach().requires_grad_(True)]

    def infer(self):
        """
        f_infer function
        compute v_0, v_1, ..., v_T and return v_T

        Returns
        -------
        v_T : torch.tensor((|facts|, ))
            valuation vector of the result of the forward-chaining inference
        """
        self.valuation_memory = [self.v_0]
        valuation = self.valuation_memory[0]

        W_softmaxed = F.softmax(torch.flatten(
            self.Ws[0]), dim=0).view(-1, len(self.X))

        for t in range(self.infer_step):
            # INFER DIC to speed up
            F_c_dic = {}
            for i in range(len(self.X)):
                F_c_dic[i] = self.F_c(i, valuation)
            b_t = torch.zeros((len(valuation), )).to(device)
            for i in range(len(self.X)):
                F_c_1 = F_c_dic[i]
                for j in range(len(self.X)):
                    F_c_2 = F_c_dic[j]
                    F_c = self.softor([F_c_1, F_c_2])
                    c_t = W_softmaxed[i][j] * F_c
                    b_t += c_t

            valuation = self.amalgamate(valuation, b_t)
            self.valuation_memory.append(valuation)

        if (valuation > 1.0).any():
            valuation = valuation / torch.max(valuation)

        return valuation
