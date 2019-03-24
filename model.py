from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        # Remember dropout rate is rate of setting neurons to 0
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        output, h_n = self.gru(x)
        output = output[:, -self.output_size:, :]
        output = self.dropout(output)
        output = self.dense(output)
        return output
