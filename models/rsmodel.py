import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class GRU(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(GRU, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.state_size, self.hidden_size, num_layers=1, batch_first=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.GRU):
            xavier_normal_(module.weight_hh_l0)
            xavier_normal_(module.weight_ih_l0)
            constant_(module.bias_ih_l0, 0)
            constant_(module.bias_hh_l0, 0)

    def forward(self, input_embeddings, state_length):
        self.gru.flatten_parameters()
        input_embeddings = nn.utils.rnn.pack_padded_sequence(input_embeddings, lengths=state_length, batch_first=True, enforce_sorted=False)
        _, state_hidden = self.gru(input_embeddings)
        return state_hidden.squeeze(0)
