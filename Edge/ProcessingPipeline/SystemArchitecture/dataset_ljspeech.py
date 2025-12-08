import torch
from torch.utils.data import Dataset
import librosa
import os

class LJSpeechDataset(Dataset):
    def __init__(self, df, root_dir, feature_extractor, phoneme_encoder, sr=22050):
        self.df = df
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.phoneme_encoder = phoneme_encoder
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.root_dir, 'wavs', row['id'] + '.wav')
        y, sr = librosa.load(wav_path, sr=self.sr)
        features = self.feature_extractor(torch.tensor(y, dtype=torch.float32))
        phonemes = self.phoneme_encoder(row['normalized_text'])
        return features, torch.tensor(phonemes, dtype=torch.long)
