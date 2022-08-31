import torch.nn as nn
import torch

from transformer import DecoderBlock, TransformerBlock
import torch.nn as nn
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn as nn
import torch

class GPT2Decoder(nn.Module):
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
        super(GPT2Decoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.pos_encoding = self.gpt2_model.transformer.wpe 
        self.embedding = self.gpt2_model.transformer.wte
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)

        ])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    
    def forward(self, enc_out, captions, src_mask, target_mask):
        N, seq_length = captions.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        input_features = self.dropout(self.embedding(captions) + self.pos_encoding(positions))

        for layer in self.layers:
            x = layer(input_features, enc_out, enc_out, src_mask, target_mask)
        
        out = self.fc_out(x)
        return out, self.gpt2_model(captions).logits, self.gpt2_model(captions).loss

class ResNetEncoder(nn.Module):
    def __init__(
        self, 
        embed_size,
        src_vocab_size,
        heads,
        device,
        dropout,
        num_layers,
        forward_expansion,
        max_length,
        model
    ):
        super(ResNetEncoder, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Linear(src_vocab_size, embed_size)
        self.pos_encoding = nn.Embedding(max_length, embed_size)
        self.device = device
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layers)
            
        ])
        self.model = model.to(self.device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, mask):
        features = self.model(features)['hidden_output']
        N, seq_length, patch_size = features.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        input_features = self.dropout(features.float() + self.pos_encoding(positions.long()))

        for layer in self.layers:
            out = layer(input_features, input_features, input_features, mask)

        return out


class VisualGPT2Transformer(nn.Module):
    def __init__(
        self, 
        encoder_model,
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
        super(VisualGPT2Transformer, self).__init__()
        self.device = device
        self.encoder = ResNetEncoder(
            embed_size,
            src_vocab_size,
            heads,
            device,
            dropout,
            num_layers,
            forward_expansion,
            max_length,
            encoder_model
        ).to(self.device)

        self.decoder = GPT2Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        ).to(self.device)
    def create_target_mask(self, target):
        N, trg_len = target.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, image, caption):
        features = self.encoder(image, None)
        target_mask = self.create_target_mask(caption)
        output, gpt2_logits, gpt2_loss = self.decoder(features, caption, None, target_mask)    
        return output, gpt2_logits, gpt2_loss