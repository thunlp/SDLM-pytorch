import torch
import time

cum_time = 0.0


class spmm(torch.autograd.Function):
    def __init__(self, weight, weight_t):
        super().__init__()
        self.weight = weight
        self.weight_t = weight_t
        self.cum_time_f = 0.0
        self.cum_time_b = 0.0

    # Note that both forward and backward are @staticmethods
    def forward(self, input):
        self.cum_time_f -= time.time()
        output = torch.mm(self.weight_t, input.t()).t()
        self.cum_time_f += time.time()
        #print('forward:{}'.format(self.cum_time_f))
        return output

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output, _indices_grad=None):
        self.cum_time_b -= time.time()
        grad_input = torch.mm(self.weight, grad_output.t()).t()
        self.cum_time_b += time.time()
        #print('backward:{}'.format(self.cum_time_b))
        return grad_input


if __name__ == '__main__':
    a = torch.autograd.Variable(torch.randn(50, 500), requires_grad=True)
    b = torch.autograd.Variable(torch.sparse.FloatTensor(500, 100))
    SPMM = spmm()
    c = SPMM(a, b).sum()
    c.backward()
