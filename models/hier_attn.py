import torch
import torch.nn as nn

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from models.attn_birnn import AttentionalBiRNN


class HAN(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, sent_attn_size, word_attn_size, num_vocab,
                 num_classes, embedd_dim, pad_idx=0, init_weight=0.1, device="cuda", vectors=None):
        super().__init__()
        self.device = device

        self.emb_lut = nn.Embedding(num_vocab, embedd_dim, pad_idx)
        if vectors is not None:
            self.emb_lut.from_pretrained(embeddings=vectors, freeze=False, padding_idx=pad_idx)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -init_weight, init_weight)

        self.word_encoder = AttentionalBiRNN(embedd_dim, word_hidden_size,
                                             word_attn_size,
                                             init_weight)

        self.sent_encoder = AttentionalBiRNN(word_hidden_size,
                                             sent_hidden_size,
                                             sent_attn_size,
                                             init_weight)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sent_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, doc_lengths, sent_lengths):
        """
        :param input_ids: (batch_size x num_sent x num_word)
        :param doc_lengths: (batch_size)
        :param sent_lengths: (batch_size x num_sent)
        """
        emb = self.emb_lut(input_ids)  # (batch_size x num_sent x num_word x emb_dim)

        packed_words = pack_padded_sequence(emb, lengths=doc_lengths.cpu().tolist(), batch_first=True)
        packed_sent_lens = pack_padded_sequence(sent_lengths, lengths=doc_lengths.cpu().tolist(), batch_first=True)

        packed_batch_size = packed_words.batch_sizes
        packed_sorted_sent_len, packed_sorted_sent_indices = torch.sort(packed_sent_lens.data, descending=True)
        packed_sorted_words = packed_words.data[packed_sorted_sent_indices]

        packed_enc_sents = self.word_encoder(packed_sorted_words, packed_sorted_sent_len)
        packed_reorder_sent_indices = torch.argsort(packed_sorted_sent_indices, descending=False)
        packed_enc_sents = packed_enc_sents[packed_reorder_sent_indices]

        packed_enc_sents = PackedSequence(packed_enc_sents, packed_batch_size)
        enc_sent, _ = pad_packed_sequence(packed_enc_sents, batch_first=True)

        doc_enc = self.sent_encoder(enc_sent, doc_lengths)

        logits = self.fc(doc_enc)
        prods = self.softmax(logits)
        return prods, logits

if __name__ == "__main__":
    input_sample = torch.randint(0, 10, (2, 3, 4))
    doc_lengths = torch.tensor([3, 2])
    sent_lengths = torch.tensor([[4, 4, 3], [4, 3, 1]])
    model = HAN(word_hidden_size=10, sent_hidden_size=10, sent_attn_size=20, word_attn_size=20, num_vocab=100,
                 num_classes=17, embedd_dim=10, pad_idx=0, init_weight=0.1, device="cuda", vectors=None)
    model(input_sample, doc_lengths, sent_lengths)