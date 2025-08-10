package MachineLearning.VoiceCloneNetwork;

public class PitchDetector {
    public static double[] detectPitch(double[] audio, int sampleRate, int windowSize, int hopSize) {
        int nFrames = (audio.length - windowSize) / hopSize + 1;
        double[] f0 = new double[nFrames];
        
        for (int t = 0; t < nFrames; t++) {
            int start = t * hopSize;
            double[] frame = new double[windowSize];
            System.arraycopy(audio, start, frame, 0, windowSize);
            
            f0[t] = yinPitchDetection(frame, sampleRate);
        }
        
        return f0;
    }
    
    private static double yinPitchDetection(double[] frame, int sampleRate) {
        int tauMax = sampleRate / 80; // Minimum expected pitch (80Hz)
        double[] difference = new double[tauMax];
        
        // Difference function
        for (int tau = 0; tau < tauMax; tau++) {
            for (int j = 0; j < tauMax; j++) {
                double delta = frame[j] - frame[j + tau];
                difference[tau] += delta * delta;
            }
        }
        
        // Cumulative mean normalized difference
        double[] cmndf = new double[tauMax];
        cmndf[0] = 1.0;
        double runningSum = 0.0;
        
        for (int tau = 1; tau < tauMax; tau++) {
            runningSum += difference[tau];
            cmndf[tau] = difference[tau] * tau / runningSum;
        }
        
        // Find first local minimum
        int tau = 2;
        while (tau < tauMax - 1 && !(cmndf[tau] < cmndf[tau-1] && cmndf[tau] < cmndf[tau+1])) {
            tau++;
        }
        
        if (tau == tauMax - 1) return 0; // Unvoiced
        
        // Parabolic interpolation for better precision
        double betterTau;
        if (tau > 0 && tau < tauMax - 1) {
            double s0 = cmndf[tau-1];
            double s1 = cmndf[tau];
            double s2 = cmndf[tau+1];
            betterTau = tau + 0.5 * (s2 - s0) / (2 * s1 - s0 - s2);
        } else {
            betterTau = tau;
        }
        
        return sampleRate / betterTau;
    }
}