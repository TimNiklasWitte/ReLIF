import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric

import tqdm

from ReLIF import *


class TimSNet(nn.Module):
    def __init__(self, input_size, n_hidden, freq_max, n_classes):
        super().__init__()

        self.num_steps = 300

        #
        # Initialize layers
        #

        self.input_modulation = self.InputModulation().apply

        self.linear_1 = nn.Linear(input_size, n_hidden)
        self.relif_1 = ReLIF(0.9, n_hidden, freq_max)

        self.linear_2 = nn.Linear(n_hidden, n_hidden)
        self.relif_2 = ReLIF(0.9, n_hidden, freq_max)

        self.linear_3 = nn.Linear(n_hidden, n_hidden)
        self.relif_3 = ReLIF(0.9, n_hidden, freq_max)

        self.linear_4 = nn.Linear(n_hidden, n_hidden)
        self.relif_4 = ReLIF(0.9, n_hidden, freq_max)

        self.linear_output = nn.Linear(n_hidden, n_classes)
        self.relif_output = ReLIF(0.9, n_classes, freq_max)


     
        self.cce_rate_loss = nn.CrossEntropyLoss()
      
        resonator_params = []
        other_params = []

        for name, p in self.named_parameters():
            if any(k in name for k in ["resonating_logit", "freq_logit", "phase_logit"]):
                resonator_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": 1e-4},
            {"params": resonator_params, "lr": 1e-2},   # much higher LR
        ])

        # , weight_decay=1e-4

        #
        # Metrics
        #

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()

  

    def forward(self, x):

        self.relif_1.reset_mem()
        self.relif_2.reset_mem()
        self.relif_3.reset_mem()
        self.relif_4.reset_mem()
        self.relif_output.reset_mem()

        #
        # Readout
        #

      
        spk_out_list = []


        x_modulated = self.temporal_smooth(x, alpha=0.9355)

        t_values = torch.linspace(0, 1, self.num_steps)

        for t in range(self.num_steps):
            
            # x: (t, bs, d) = (300, 64, 4624)

     
            x_t = self.linear_1(x[t, ...])
            x_t, _ = self.relif_1(t_values[t], x_t)

            x_t = self.linear_2(x_t)
            x_t, _ = self.relif_2(t_values[t], x_t)

            x_t = self.linear_3(x_t)
            x_t, _ = self.relif_3(t_values[t], x_t)

            x_t = self.linear_4(x_t)
            x_t, _ = self.relif_4(t_values[t], x_t)

            spikes = self.linear_output(x_t)
         
            spikes_output, _ = self.relif_output(t_values[t], spikes)


            spk_out_list.append(spikes_output)

            
        spk_rec = torch.stack(spk_out_list, dim=0)
      

        return spk_rec

    
    @torch.no_grad
    def test(self, test_loader, device):

        self.eval()

        self.loss_metric.reset()
        self.accuracy_metric.reset()


        for x, targets in tqdm.tqdm(test_loader, position=0, leave=True):
            
            # x: (bs, t, d) = (64, 20, 1156)
            x = x.permute(1,0,2)

            # x: (t, bs, d) = (20, 64, 1156)

            # Transfer data to GPU (if available)
            x, targets = x.to(device), targets.to(device)

            # Forward pass
            spk_rec = self(x)
            spk_sum = spk_rec.sum(dim=0)
        

            loss = self.cce_rate_loss(spk_sum, targets)

            #
            # Update metrics
            #

            # Loss
            self.loss_metric.update(loss)

            # Accuracy
            accuracy = SF.accuracy_rate(spk_rec, targets)
            self.accuracy_metric.update(accuracy)

    
        
        test_loss = self.loss_metric.compute()
        test_accuracy = self.accuracy_metric.compute()
        
     
    
        return test_loss, test_accuracy
    

    class InputModulation(torch.autograd.Function):

        def __init__(self):
            super().__init__()

            self.decay = 0.9355

        @staticmethod
        def forward(ctx, x_t, x_t_modulated):
      
            ctx.save_for_backward(x_t_modulated)
            return x_t

        @staticmethod
        def backward(ctx, grad_output):
            (x_t_modulated,) = ctx.saved_tensors
 
            grad = x_t_modulated * grad_output
            
            return grad
