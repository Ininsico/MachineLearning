import librosa
import matplotlib.pyplot as plt
import numpy as np

wavepath = 'test.wav'
y, sr = librosa.load(wavepath,sr=None)
chorma = librosa.feature.chroma_stft(y=y,sr=sr)
plt.figure(figsize=(14,5))
librosa.display.specshow(chorma, y_axis='chroma',x_axis='time')
plt.colorbar()
plt.title('Somethin Chorma')
plt.show()
