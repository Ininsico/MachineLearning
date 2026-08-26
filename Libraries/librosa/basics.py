import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
audiopath = 'test.wav'
y, sr = librosa.load(audiopath, sr=None)

# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=sr)
# plt.title('Audio Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(14,5))
librosa.display.specshow(D, y_axis='log',x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrograph')
plt.show()