import torch
import torch.nn as nn

class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        nn.init.orthogonal_(self.h2h.weight)
        self.h2out = nn.Linear(hidden_size, output_size)
        self.initial_hidden = None

    def init_hidden(self, input_shape):
        init = torch.zeros(self.hidden_size)
        self.initial_hidden = init
        return init

    def recurrence(self, input, hidden):
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape)

        # Loop through time
        hidden_all = []
        steps = range(input.size(0))
        for s in steps:
            hidden = self.recurrence(input[s], hidden)
            hidden_all.append(hidden)

        # Stack together output from all time steps
        hidden_all = torch.stack(hidden_all, dim=0) 
        output = self.h2out(hidden_all)

        return output, hidden_all