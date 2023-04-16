import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Linear(input_size, hidden_size)

        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.num_layers)]
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        self.hidden = []

        for i in range(self.num_layers):
            self.hidden.append(
                (
                    torch.zeros(batch_size, self.hidden_size).to(device),  # h
                    torch.zeros(batch_size, self.hidden_size).to(device),  # c
                )
            )

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))

        h_in = embedded

        for i in range(self.num_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])

            h_in = self.hidden[i][0]

        output = self.fc(h_in)

        return output


class ProbabilisticLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(ProbabilisticLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Linear(input_size, hidden_size)

        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.num_layers)]
        )

        self.mu_network = nn.Linear(hidden_size, output_size)
        self.logvar_network = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        self.hidden = []

        for i in range(self.num_layers):
            self.hidden.append(
                (
                    torch.zeros(batch_size, self.hidden_size).to(device),  # h
                    torch.zeros(batch_size, self.hidden_size).to(device),  # c
                )
            )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))

        h_in = embedded

        for i in range(self.num_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])

            h_in = self.hidden[i][0]

        mu = self.mu_network(h_in)
        logvar = self.logvar_network(h_in)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
