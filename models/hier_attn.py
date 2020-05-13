import torch
import torch.nn as nn

from models.attn_birnn import AttentionalBiRNN


class HAN(nn.Module):
    def __init__(self, word_hidden_size, sentence_hidden_size, sentence_attn_size, word_attn_size, num_vocab,
                 num_classes, embedd_dim, pad_idx=0, init_weight=0.1, device="cuda", vectors=None):
        super().__init__()
        self.device = device

        self.emb_lut = nn.Embedding(num_vocab, embedd_dim, pad_idx)
        if vectors:
            self.emb_lut.from_pretrained(embeddings=vectors, freeze=False)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -init_weight, init_weight)

        self.word_encoder = AttentionalBiRNN(embedd_dim, word_hidden_size,
                                             word_attn_size,
                                             init_weight)

        self.sent_encoder = AttentionalBiRNN(word_hidden_size,
                                             sentence_hidden_size,
                                             sentence_attn_size,
                                             init_weight)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sentence_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, sent_lengths, doc_lengths):
        emb = self.emb_lut(input_ids)
        emb = emb.permute(1, 0, 2, 3)
        sent_lengths = sent_lengths.permute(1, 0)
        enc_words = []
        for sent, len in list(zip(emb, sent_lengths)):
            enc_word = self.word_encoder(sent, len)
            enc_words.append(enc_word.unsqueeze(0))
        enc_words = torch.cat(enc_words, dim=0)
        enc_words = enc_words.permute(1, 0, 2)
        enc_words = self.dropout(enc_words)
        enc_sents = self.sent_encoder(enc_words, doc_lengths)
        logits = self.fc(enc_sents)
        prods = self.softmax(logits)
        return prods, logits


if __name__ == "__main__":
    input_sample = torch.randint(0, 10, (2, 3, 4))
    sent_lengths = torch.tensor([[4, 4, 3], [4, 3, 1]])
    doc_lengths = torch.tensor([3, 2])
    model = HAN(word_hidden_size=10, sentence_hidden_size=10, sentence_attn_size=20, word_attn_size=20, num_vocab=100,
                 num_classes=17, embedd_dim=10, pad_idx=0, init_weight=0.1, device="cuda", vectors=None)
    model(input_sample, sent_lengths, doc_lengths)