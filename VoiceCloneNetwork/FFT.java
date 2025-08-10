package MachineLearning.VoiceCloneNetwork;

public class FFT {
    public static double[] fft(double[] input) {
        int n = input.length;
        double[] output = new double[n * 2];
        
        // Copy real input to complex array (imaginary part = 0)
        for (int i = 0; i < n; i++) {
            output[2*i] = input[i];
            output[2*i+1] = 0;
        }
        
        fft(output, n);
        return output;
    }
    
    private static void fft(double[] buffer, int n) {
        if (n <= 1) return;
        
        // Bit-reversal permutation
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (j > i) {
                swap(buffer, 2*i, 2*j);
                swap(buffer, 2*i+1, 2*j+1);
            }
            
            int m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
        
        // Danielson-Lanczos algorithm
        for (int s = 1; s <= Math.log(n)/Math.log(2); s++) {
            int m = 1 << s;
            double wReal = 1.0;
            double wImag = 0.0;
            double theta = -2 * Math.PI / m;
            double wTempReal = Math.cos(theta);
            double wTempImag = Math.sin(theta);
            
            for (int o = 0; o < m/2; o++) {
                for (int k = o; k < n; k += m) {
                    int t = k + m/2;
                    double tReal = wReal * buffer[2*t] - wImag * buffer[2*t+1];
                    double tImag = wReal * buffer[2*t+1] + wImag * buffer[2*t];
                    
                    buffer[2*t] = buffer[2*k] - tReal;
                    buffer[2*t+1] = buffer[2*k+1] - tImag;
                    buffer[2*k] += tReal;
                    buffer[2*k+1] += tImag;
                }
                
                double tempReal = wReal * wTempReal - wImag * wTempImag;
                double tempImag = wReal * wTempImag + wImag * wTempReal;
                wReal = tempReal;
                wImag = tempImag;
            }
        }
    }
    
    private static void swap(double[] buffer, int i, int j) {
        double temp = buffer[i];
        buffer[i] = buffer[j];
        buffer[j] = temp;
    }
}