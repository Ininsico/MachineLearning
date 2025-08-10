package MachineLearning.VoiceCloneNetwork;

public class MelSpectrogram {
    private static final double MIN_FREQ = 80.0;
    private static final double MAX_FREQ = 8000.0;
    
    public static double[][] computeMelSpectrogram(double[] audio, int sampleRate, 
            int windowSize, int hopSize, int nMelBins) {
        
        int nFrames = (audio.length - windowSize) / hopSize + 1;
        double[][] melSpectrogram = new double[nMelBins][nFrames];
        
        // Pre-compute Mel filter banks
        double[][] melFilters = createMelFilterBank(sampleRate, windowSize, nMelBins);
        
        // Process each frame
        for (int t = 0; t < nFrames; t++) {
            int start = t * hopSize;
            double[] frame = new double[windowSize];
            System.arraycopy(audio, start, frame, 0, windowSize);
            
            // Apply window function (Hamming)
            applyWindow(frame);
            
            // Compute FFT magnitude spectrum
            double[] spectrum = computePowerSpectrum(frame);
            
            // Apply Mel filter banks
            for (int m = 0; m < nMelBins; m++) {
                double melEnergy = 0.0;
                for (int k = 0; k < spectrum.length; k++) {
                    melEnergy += spectrum[k] * melFilters[m][k];
                }
                melSpectrogram[m][t] = 10 * Math.log10(Math.max(melEnergy, 1e-10));
            }
        }
        
        return melSpectrogram;
    }
    
    private static void applyWindow(double[] frame) {
        for (int i = 0; i < frame.length; i++) {
            frame[i] *= 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (frame.length - 1));
        }
    }
    
    private static double[] computePowerSpectrum(double[] frame) {
        // Implement FFT here (see next section)
        double[] fft = FFT.fft(frame);
        double[] spectrum = new double[fft.length / 2];
        for (int i = 0; i < spectrum.length; i++) {
            double re = fft[2*i];
            double im = fft[2*i+1];
            spectrum[i] = re*re + im*im;
        }
        return spectrum;
    }
    
    private static double[][] createMelFilterBank(int sampleRate, int fftSize, int nMelBins) {
        double[][] filters = new double[nMelBins][fftSize/2 + 1];
        
        double minMel = hzToMel(MIN_FREQ);
        double maxMel = hzToMel(MAX_FREQ);
        
        double[] melPoints = new double[nMelBins + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = minMel + i * (maxMel - minMel) / (nMelBins + 1);
        }
        
        double[] hzPoints = new double[melPoints.length];
        for (int i = 0; i < hzPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }
        
        int[] binIndices = new int[hzPoints.length];
        for (int i = 0; i < binIndices.length; i++) {
            binIndices[i] = (int) Math.floor(hzPoints[i] * fftSize / sampleRate);
        }
        
        for (int m = 1; m <= nMelBins; m++) {
            int left = binIndices[m-1];
            int center = binIndices[m];
            int right = binIndices[m+1];
            
            for (int k = left; k < center; k++) {
                filters[m-1][k] = (k - left) / (double) (center - left);
            }
            for (int k = center; k < right; k++) {
                filters[m-1][k] = (right - k) / (double) (right - center);
            }
        }
        
        return filters;
    }
    
    private static double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700.0);
    }
    
    private static double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }
}
