import torch
import torch.nn as nn



class ReLIF(nn.Module):
 

    def __init__(self, beta, n_hidden, freq_max):
        super().__init__()

        self.beta = beta

        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)

        #
        # Resonator
        #

        self.resonating_logit = nn.Parameter(
            torch.empty(n_hidden).uniform_(-2, 2)
        )

        self.freq_max = freq_max
        self.freq_logit = nn.Parameter(
            torch.empty(n_hidden).uniform_(-2, 2)
        )

        self.phase_logit = nn.Parameter(
            torch.empty(n_hidden).uniform_(-2, 2)
        )



        self.threshold = 1.0

        self.spike_gradient = self.SigmoidSurrogate.apply

    def forward(self, t, input_, mem=None):

        if not mem == None:
            self.mem = mem

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)


        #
        # Resonator
        #

        freq = self.freq_max * torch.sigmoid(self.freq_logit)
        phase = 2 * torch.pi * torch.sigmoid(self.phase_logit)

        resonating = torch.sigmoid(self.resonating_logit)

        resonator_activation = 0.5 * resonating * torch.sin(
            2 * torch.pi * freq * t + phase
        )


        #
        # LIF
        #

        spk = self.spike_gradient(self.mem, self.threshold)


        reset = (self.beta * spk * self.threshold).detach()
        self.mem = self.beta * self.mem + input_ + resonator_activation- reset

        return spk.float(), self.mem
    

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    class SigmoidSurrogate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v, threshold):
            ctx.save_for_backward(v)
            ctx.threshold = threshold

            return (v >= threshold).float()

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors

            x = v - ctx.threshold
            k = 1

            grad = 1.0 / (1.0 + (k * x.abs()) ** 2)

            return grad_output * grad, None
