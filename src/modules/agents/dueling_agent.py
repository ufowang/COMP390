import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DuelingAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DuelingAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc_value = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc_advantage = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        value = self.fc_value(h)
        advantage = self.fc_advantage(h)
        q = value + advantage - advantage.max(dim=-1, keepdim=True).values
        return q, h
