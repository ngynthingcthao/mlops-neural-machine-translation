import torch
import torch.nn as nn
import math
# Embedding + Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class TransformerTranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers,
                                          dim_feedforward, dropout, batch_first=True)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_emb = self.pos_encoder(self.src_tok_emb(src))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))

        output = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_pad_mask,
                                tgt_key_padding_mask=tgt_pad_mask,
                                memory_key_padding_mask=src_pad_mask)
        return self.fc_out(output)
