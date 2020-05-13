import torch
import torch.nn as nn

from models.common import init_lstm_


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