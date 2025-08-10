package MachineLearning.VoiceCloneNetwork;

public class AperiodicityDetector {
    public static double[] detectAperiodicity(double[] audio, double[] f0, 
            int sampleRate, int windowSize, int hopSize) {
        
        int nFrames = f0.length;
        double[] aperiodicity = new double[nFrames];
        
        for (int t = 0; t < nFrames; t++) {
            int start = t * hopSize;
            double[] frame = new double[windowSize];
            System.arraycopy(audio, start, frame, 0, windowSize);
            
            if (f0[t] <= 0) {
                aperiodicity[t] = 1.0; // Fully unvoiced
            } else {
                aperiodicity[t] = computeHNR(frame, sampleRate, f0[t]);
            }
        }
        
        return aperiodicity;
    }
    
    private static double computeHNR(double[] frame, int sampleRate, double f0) {
        int period = (int) (sampleRate / f0);
        double autoCorr = 0;
        double energy = 0;
        
        for (int i = 0; i < frame.length - period; i++) {
            autoCorr += frame[i] * frame[i + period];
            energy += frame[i] * frame[i];
        }
        
        if (energy < 1e-10) return 1.0;
        
        double hnr = autoCorr / (energy - autoCorr + 1e-10);
        // Convert to 0-1 range where 0=voiced, 1=unvoiced
        return 1.0 / (1.0 + hnr);
    }
}