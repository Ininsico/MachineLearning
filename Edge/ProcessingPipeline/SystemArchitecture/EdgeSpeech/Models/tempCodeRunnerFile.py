import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
EMB_DIM = 512
HIDDEN_DIM = 512
CNN_CHANNELS = 512
NUM_LAYERS = 4
N_HEADS = 8
DROPOUT = 0.2
BATCH_SIZE = 64 if torch.cuda.is_available() else 64
LABEL_SMOOTHING = 0.1
MAX_LR = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
GRAD_CLIP = 1.0
class G2PInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        
        # Reconstruct model architecture
        self.model = MegaG2P(
            char_vocab_size=len(self.config['char2idx']),
            phon_vocab_size=len(self.config['phon2idx'])
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Vocabularies
        self.char2idx = self.config['char2idx']
        self.idx2char = {v:k for k,v in self.char2idx.items()}
        self.phon2idx = self.config['phon2idx']
        self.idx2phon = self.config['idx2phon']
        
        # Special tokens
        self.sos_token = self.phon2idx['<sos>']
        self.eos_token = self.phon2idx['<eos>']
        self.pad_token = 0
    
    def preprocess(self, text):
        """Convert text to tensor"""
        text = text.lower().strip()
        chars = [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]
        return torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def postprocess(self, phon_ids):
        """Convert phoneme IDs to string"""
        phons = []
        for idx in phon_ids:
            if idx == self.eos_token:
                break
            if idx != self.sos_token and idx != self.pad_token:
                phons.append(self.idx2phon[idx])
        return ' '.join(phons)
    
    def predict(self, text, max_len=100):
        """Predict phonemes for input text"""
        with torch.no_grad():
            # Prepare input
            src = self.preprocess(text)
            src_len = torch.tensor([src.size(1)], device=self.device)
            
            # Encode
            memory = self.model.encoder(src, src_len)
            
            # Initialize with SOS token
            tgt = torch.ones(1, 1).fill_(self.sos_token).long().to(self.device)
            
            # Generate tokens one by one
            for _ in range(max_len):
                logits, _ = self.model.decoder(
                    tgt, 
                    memory,
                    memory_key_padding_mask=(src == self.pad_token)
                )
                next_token = logits[:, -1, :].argmax(-1).unsqueeze(-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.eos_token:
                    break
            
            # Convert to phonemes
            phon_ids = tgt.squeeze().cpu().numpy()
            return self.postprocess(phon_ids)

# You need to include these model definitions from your training code
class MegaG2P(nn.Module):
    def __init__(self, char_vocab_size, phon_vocab_size):
        super().__init__()
        self.encoder = CNNEncoder(char_vocab_size)
        self.decoder = TransformerDecoder(phon_vocab_size)
    
    def forward(self, src, src_lens, tgt):
        memory = self.encoder(src, src_lens)
        dec_input = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1)).to(src.device)
        logits, ctc_logits = self.decoder(
            dec_input,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(src == 0)
        )
        return logits, ctc_logits

# Include these if not already defined
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMB_DIM, DROPOUT)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=5, padding=2),
                nn.BatchNorm1d(EMB_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT)
            ) for _ in range(3)
        ])
        encoder_layers = TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=N_HEADS,
            dim_feedforward=EMB_DIM*4,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=2)

    def forward(self, src, src_lens):
        embedded = self.embedding(src) * math.sqrt(EMB_DIM)
        embedded = self.pos_encoder(embedded)
        conv_out = embedded.permute(0, 2, 1)
        for conv in self.conv_layers:
            conv_out = conv(conv_out) + conv_out
        conv_out = conv_out.permute(0, 2, 1)
        src_key_padding_mask = (src == 0)
        transformer_out = self.transformer_encoder(conv_out, src_key_padding_mask=src_key_padding_mask)
        return transformer_out

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMB_DIM, DROPOUT)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=EMB_DIM,
                nhead=N_HEADS,
                dim_feedforward=EMB_DIM*4,
                dropout=DROPOUT,
                batch_first=True
            ) for _ in range(NUM_LAYERS)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.GELU(),
            nn.LayerNorm(EMB_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(EMB_DIM, vocab_size)
        )
        self.ctc_proj = nn.Linear(EMB_DIM, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        embedded = self.embedding(tgt) * math.sqrt(EMB_DIM)
        embedded = self.pos_encoder(embedded)
        output = embedded
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        logits = self.output_proj(output)
        ctc_logits = self.ctc_proj(output).log_softmax(dim=-1)
        return logits, ctc_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# How to use it
if __name__ == '__main__':
    # Initialize with your downloaded model
    g2p = G2PInference('g2p_megatron.pth')  # Make sure the file is in the same directory
    
    # Test some words
    test_words = ["hello", "world", "artificial", "intelligence", "testing"]
    
    for word in test_words:
        phonemes = g2p.predict(word)
        print(f"{word.upper():<15} → {phonemes}")
    
    # Or interactive mode
    print("\nEnter words to get phonemes (type 'quit' to exit):")
    while True:
        word = input("> ").strip()
        if word.lower() in ['quit', 'exit']:
            break
        print(g2p.predict(word))