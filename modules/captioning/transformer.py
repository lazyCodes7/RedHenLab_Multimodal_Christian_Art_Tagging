import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size/self.heads

        assert(self.head_dim*self.heads == embed_size) , "Embed size need to be div by heads"

        self.keys = nn.Linear(head_dim, head_dim)
        self.values = nn.Linear(head_dim, head_dim)
        self.queries = nn.Linear(head_dim, head_dim)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask = 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.forward_expansion = forward_expansion
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
    
    def forward(self, values, keys, query, mask):
        attention = self.attention(values, keys, query, mask)
        attention_out = self.dropout(self.norm1(attention + query))
        ffn = self.ffn(attention_out)
        ffn_out = self.dropout(self.norm2(ffn+attention_out))
        return ffn_out


class Encoder(nn.Module):
    def __init__(
        self, 
        embed_size,
        src_vocab_size,
        heads,
        device,
        dropout,
        num_layers,
        forward_expansion,
        max_length
    ):
        self.embed_size = embed_size
        self.embedding = nn.Embedding(src_vocab_size, embed_size)
        self.pos_encoding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layers)
            
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, mask):
        N, seq_length = features.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        input_features = self.dropout(self.embedding(features) + self.pos_encoding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device
    ):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self, captions, value, key, src_mask, target_mask):
        attention = self.attention(captions, captions, captions, target_mask)
        query = self.dropout(self.norm(attention + captions))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.pos_encoding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)

        ])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, enc_out, captions, src_mask, target_mask):
        N, seq_length = captions.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        input_features = self.dropout(self.embedding(captions) + self.pos_encoding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        target_pad_idx,
        embed_size = 512,
        num_layers = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0.2,
        device = 'cuda',
        max_length = 200
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            src_vocab_size,
            heads,
            device,
            dropout,
            num_layers,
            forward_expansion,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def create_target_mask(self, target):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, features, captions):
        enc_out = self.encoder(features)
        trg_mask = self.create_target_mask(captions)
        out = self.decoder(enc_out, captions,  None, trg_mask)
        return out






