import librosa 
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

audiopath = 'test.wav'
y, sr = librosa.load(audiopath, sr=None)
mfcss = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(14,5))
librosa.display.specshow(mfcss, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()
