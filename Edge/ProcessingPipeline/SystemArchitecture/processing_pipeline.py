import pandas as pd
import torch
from torch.utils.data import DataLoader
from mel_spectrogram import Melspectrogram
from phoneme_encoder import PhonemeEncoder
from dataset_ljspeech import LJSpeechDataset

# Load metadata
metadata_path = r"C:\Users\arsla\Downloads\archive (4)\LJSpeech-1.1\metadata.csv"
df = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'text', 'normalized_text'])

# Initialize components
feature_extractor = Melspectrogram(n_mels=80)
phoneme_encoder = PhonemeEncoder()

# Create dataset
dataset = LJSpeechDataset(
    df=df,
    root_dir=r"C:\Users\arsla\Downloads\archive (4)\LJSpeech-1.1",
    feature_extractor=feature_extractor,
    phoneme_encoder=phoneme_encoder
)

# DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Quick test
for mel, phonemes in loader:
    print("Mel spectrogram shape:", mel.shape)
    print("Phoneme IDs:", phonemes)
    break
