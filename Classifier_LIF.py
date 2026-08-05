import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from torchmetrics import MeanMetric

import tqdm

from LIF import *


class Classifier_LIF(nn.Module):
    def __init__(self, input_size, n_hidden, n_classes):
        super().__init__()

        self.num_steps = 300

        #
        # Initialize layers
        #

        self.input_modulation = self.InputModulation().apply

        self.linear = nn.Linear(input_size, n_hidden)
        self.relif_1 = LIF(0.9, n_hidden)


        self.linear_output = nn.Linear(n_hidden, n_classes)
        self.relif_2 = LIF(0.9, n_classes)


     
        self.cce_rate_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)


        #
        # Metrics
        #

        self.loss_metric = MeanMetric()
        self.accuracy_metric = MeanMetric()

    # input modulation
    def temporal_smooth(self, x, alpha):
        # x: (t, bs, d)
        y = x.clone()
        
        for t in range(1, x.size(0)):
            y[t] = alpha * y[t-1]
        
        return y

    def forward(self, x):
        
        batch_size = x.shape[1]

        self.relif_1.reset_mem()
        self.relif_2.reset_mem()

        #
        # Readout
        #

      
        spk_out_list = []


        x_modulated = self.temporal_smooth(x, alpha=0.9355)

        t_values = torch.linspace(0, 1, self.num_steps)

        for t in range(self.num_steps):
            
            # x: (t, bs, d) = (300, 64, 4624)

            x_t = self.input_modulation(x[t, ...], x_modulated[t, ...])

            x_t = self.linear(x_t)
            
            spikes, _ = self.relif_1(t_values[t], x_t)

            spikes = self.linear_output(spikes)
         
            spikes_output, _ = self.relif_2(t_values[t], spikes)


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