import torch
import torch.nn as nn

def init_lstm_(lstm, init_weight=0.1):
    """
    Initializes weights of LSTM layer.
    Weights and biases are initialized with uniform(-init_weight, init_weight)
    distribution.
    :param lstm: instance of torch.nn.LSTM
    :param init_weight: range for the uniform initializer
    """
    # Initialize hidden-hidden weights
    nn.init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    # Initialize input-hidden weights:
    nn.init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)

    # Initialize bias. PyTorch LSTM has two biases, one for input-hidden GEMM
    # and the other for hidden-hidden GEMM. Here input-hidden bias is
    # initialized with uniform distribution and hidden-hidden bias is
    # initialized with zeros.
    nn.init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    nn.init.zeros_(lstm.bias_hh_l0.data)

    if lstm.bidirectional:
        nn.init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
        nn.init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)

        nn.init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        nn.init.zeros_(lstm.bias_hh_l0_reverse.data)


class AttentionalBiRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim, init_weight):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=(hidden_dim//2),
                          bidirectional=True, batch_first=True)
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.V = nn.Parameter(torch.randn(attn_dim).float())
        self.softmax = nn.Softmax(dim=-1)
        # Init GRU's weight
        init_lstm_(self.gru, init_weight)

    def forward(self, inputs, lengths):
        packed_batch = nn.utils.rnn.pack_padded_sequence(inputs, lengths=lengths.cpu().tolist(), batch_first=True)
        last_outputs, _ = self.gru(packed_batch)
        encode, len_s = torch.nn.utils.rnn.pad_packed_sequence(last_outputs, batch_first=True)

        Hw = torch.tanh(self.W(encode))
        w_score = self.softmax(Hw.matmul(self.V))
        encode = encode.mul(w_score.unsqueeze(-1))
        encode = torch.sum(encode, dim=1)
        return encode


if __name__ == "__main__":
    input_sample = torch.rand((2, 4, 300))
    length_sample = torch.tensor([4, 2])
    attn = AttentionalBiRNN(300, 512, 1024, 0.1)
    attn(input_sample, length_sample)
