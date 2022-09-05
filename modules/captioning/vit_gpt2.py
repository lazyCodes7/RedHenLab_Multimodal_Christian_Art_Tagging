import torch.nn as nn
import torch

from transformer import DecoderBlock, TransformerBlock
import torch.nn as nn
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn as nn
import torch

class GPT2Decoder(nn.Module):
    '''
    About:
        Instantiates a GPT2Decoder block of the VisualGPT2Transformer

    Inputs:
        target_vocab_size - Int - Expects the size of the word vocabulary used while tokenizing
        embed_size - Int - Size of the lookup table
        num_layers - Int - No of layer of stacked transformer block
        heads - Int - Parameter for no of attention heads
        forward_expansion - Int - For expanding and contracting the generated outputs
        dropout - Float - Probablity to drop some nodes
        device - String - 'cuda/cpu' - Device to work with
    
    
    Methods:
        1. forward(self, enc_out, captions, src_mask, target_mask)    

            About:
                Implementation of the forward propagation for the GPT2 Decoder
            
            Inputs:
                enc_out -> Output of the Transformer's encoder block
                captions -> Tokenized captions
                src_mask -> For masking the encoder outputs
                target_mask -> For masking the decoder inputs

            Outputs:
                captions_rep -> (N, target_vocab_size) - Intermediate Representation of the Caption
                logits -> (N, target_vocab_size) - Outputs from GPT2  model
                loss -> Loss from the GPT2 model
    
    Example:
        from vit_gpt2 import GPT2Decoder
        decoder = GPT2Decoder(
            target_vocab_size = 50257,
            embed_size = 768,
            num_layers = 6,
            forward_expansion = 4,
            heads = 8,
            dropout = 0.2,
            device = device,
        )

        generated = decoder(enc_out, captions, None, target_mask)
        print(generated[0].argmax(dim=1)) # This will be the tokenized captions that we predict
        
    '''
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


        ## Pretrained GPT2Tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

        ## Adding positional encoding as in the transformer
        self.pos_encoding = self.gpt2_model.transformer.wpe 
        self.embedding = self.gpt2_model.transformer.wte
        self.dropout = nn.Dropout(dropout)

        ## Transformer layers
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

class VitEncoder(nn.Module):
    def __init__(
        self, 
        embed_size,
        src_vocab_size,
        heads,
        device,
        dropout,
        num_layers,
        forward_expansion,
        model
    ):
        '''
            About:
                Instantiates the encoder for VisualGPT2Transformer
            
            Inputs:
                src_vocab_size - Int - Expects the size of the features used in feature extractor
                embed_size - Int - Size of the lookup table
                num_layers - Int - No of layer of stacked transformer block
                heads - Int - Parameter for no of attention heads
                forward_expansion - Int - For expanding and contracting the generated outputs
                dropout - Float - Probablity to drop some nodes
                device - String - 'cuda/cpu' - Device to work with,
                model - nn.Module - Pretrained feature extractor
            
            Example:
                from vit_gpt2 import ViTEncoder
                encoder = ViTEncoder(
                    src_vocab_size = 512,
                    embed_size = 768,
                    heads = 9,
                    device = 'cuda',
                    dropout = 0.4,
                    num_layers = 6,
                    forward_expansion = 4,
                    model = model
                )
                enc_out = encoder(features = image)
                # enc_out is used by the decoder in GPT2Decoder
            
            Methods:
                1. forward(self, features, mask = None)
                    About:
                        Implementation of the forward propagation for the GPT2 Decoder

                    Inputs:
                        features - Images that are to be used by the pretrained by the pretrained feature extractor
                        mask - Optional mask to make the encoder see only parts of the features
                    
                    Outputs:
                        out - Intermediate representation/embedding to be used by the decoder block for captioning.
                                        
        '''
        super(ViTEncoder, self).__init__()
        self.embed_size = embed_size
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
    
    def forward(self, features, mask = None):
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
        embed_size = 512,
        num_layers = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0.2,
        device = 'cuda',
        max_length = 200
    ):
        '''
        About: 
            Instantiates the VisualGPT2Transformer
        
        Inputs:
            encoder_model - nn.Module - Pretrained feature extractor,
            src_vocab_size - Int - Expects the size of the features used in feature extractor
            embed_size - Int - Size of the lookup table
            num_layers - Int - No of layer of stacked transformer block
            heads - Int - Parameter for no of attention heads
            forward_expansion - Int - For expanding and contracting the generated outputs
            dropout - Float - Probablity to drop some nodes
            device - String - 'cuda/cpu' - Device to work with
            model - nn.Module - Pretrained feature extractor
            max_length - Int - Output of the captions
            target_vocab_size - Int - Expects the size of the word vocabulary used while tokenizing
        
        Methods:

            1. create_target_mask(self, target)

                About:
                    Function to create a mask for decoder
                
                Inputs:
                    target - tokenized caption to create mask
                
                Outputs:
                    trg_mask - Mask to hide the subsequent word at each point
            
            2. forward(self, image, caption)

                About:
                    Implementation of the forward propagation for the VisualGPT2Transformer

                Inputs:
                    image - Image to pass to encoder
                    caption - tokenized captions to be passed to be decoder along with enc_out
                
                Outputs:
                    Logits - to compute the caption words and loss
                    Loss - GPT2 loss to further improve performance
        
        Example:
            from vit_gpt2 import VisualGPT2Transformer
            model = VisualGPT2Transformer(
                    encoder_model,
                    src_vocab_size,
                    target_vocab_size,
                    embed_size = 512,
                    num_layers = 6,
                    forward_expansion = 4,
                    heads = 8,
                    dropout = 0.2,
                    device = 'cuda',
                    max_length = 200
            )
            outputs = model(image, caption)
            print(outputs[0].argmax(dim=1)) # Predicted tokenized caption

        '''
        super(VisualGPT2Transformer, self).__init__()
        self.device = device
        self.encoder = ViTEncoder(
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