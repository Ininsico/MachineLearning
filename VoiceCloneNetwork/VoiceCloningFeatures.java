package MachineLearning.VoiceCloneNetwork;

public class VoiceCloningFeatures {
    public static void main(String[] args) throws Exception {
        // Parameters
        int sampleRate = 24000;
        int windowSize = (int) (0.025 * sampleRate); // 25ms
        int hopSize = (int) (0.010 * sampleRate);    // 10ms
        int nMelBins = 80;
        
        // Read WAV file
        double[] audio = WavReader.readWav("input.wav");
        
        // Extract features
        double[][] melSpectrogram = MelSpectrogram.computeMelSpectrogram(
            audio, sampleRate, windowSize, hopSize, nMelBins);
        
        double[] f0 = PitchDetector.detectPitch(
            audio, sampleRate, windowSize, hopSize);
        
        double[] aperiodicity = AperiodicityDetector.detectAperiodicity(
            audio, f0, sampleRate, windowSize, hopSize);
        
        // Now you have all features for your voice cloning model
        System.out.println("Feature extraction complete!");
    }
}
