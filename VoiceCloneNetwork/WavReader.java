package MachineLearning.VoiceCloneNetwork;

import java.io.*;
import javax.sound.sampled.*;
import java.io.*;
import javax.sound.sampled.*;

public class WavReader {
    public static double[] readWav(String filePath) throws UnsupportedAudioFileException, IOException {
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
        AudioFormat format = audioInputStream.getFormat();
        
        if (format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED) {
            throw new UnsupportedAudioFileException("Only PCM_SIGNED encoding supported");
        }
        
        byte[] audioBytes = audioInputStream.readAllBytes();
        int sampleSize = format.getSampleSizeInBits() / 8;
        double[] samples = new double[audioBytes.length / sampleSize];
        
        for (int i = 0; i < samples.length; i++) {
            int byteIndex = i * sampleSize;
            int value = 0;
            
            // Read little-endian samples
            for (int b = 0; b < sampleSize; b++) {
                value |= (audioBytes[byteIndex + b] & 0xFF) << (8 * b);
            }
            
            // Normalize to [-1, 1]
            samples[i] = value / (double) (1 << (8 * sampleSize - 1));
        }
        
        return samples;
    }
}