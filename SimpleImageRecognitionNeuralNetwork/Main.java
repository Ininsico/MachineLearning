import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
import java.awt.Graphics2D;

public class Main {

    // Custom Matrix class with all operations
    static class Matrix {
        double[][] data;
        int rows, cols;

        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.data = new double[rows][cols];
        }

        public void randomize(double min, double max) {
            Random rand = new Random();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = min + (max - min) * rand.nextDouble();
                }
            }
        }

        public static Matrix multiply(Matrix a, Matrix b) {
            Matrix result = new Matrix(a.rows, b.cols);
            for (int i = 0; i < result.rows; i++) {
                for (int j = 0; j < result.cols; j++) {
                    double sum = 0;
                    for (int k = 0; k < a.cols; k++) {
                        sum += a.data[i][k] * b.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
            return result;
        }

        public static Matrix add(Matrix a, Matrix b) {
            Matrix result = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] + b.data[i][j];
                }
            }
            return result;
        }

        public static Matrix subtract(Matrix a, Matrix b) {
            Matrix result = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] - b.data[i][j];
                }
            }
            return result;
        }

        public static Matrix transpose(Matrix a) {
            Matrix result = new Matrix(a.cols, a.rows);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[j][i] = a.data[i][j];
                }
            }
            return result;
        }

        public static Matrix hadamard(Matrix a, Matrix b) {
            Matrix result = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] * b.data[i][j];
                }
            }
            return result;
        }

        public static Matrix scalarMultiply(Matrix a, double scalar) {
            Matrix result = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] * scalar;
                }
            }
            return result;
        }

        public void applyFunction(ActivationFunction func) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = func.activate(data[i][j]);
                }
            }
        }

        public Matrix copy() {
            Matrix result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                System.arraycopy(data[i], 0, result.data[i], 0, cols);
            }
            return result;
        }
    }

    interface ActivationFunction {
        double activate(double x);

        double derivative(double x);
    }

    static class LeakyReLU implements ActivationFunction {
        public double activate(double x) {
            return x > 0 ? x : 0.01 * x;
        }

        public double derivative(double x) {
            return x > 0 ? 1 : 0.01;
        }
    }

    static class Sigmoid implements ActivationFunction {
        public double activate(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        public double derivative(double x) {
            double s = activate(x);
            return s * (1 - s);
        }
    }

    // Face Generation Autoencoder
    static class FaceGenerator {
        // Network architecture
        List<Matrix> encoderWeights = new ArrayList<>();
        List<Matrix> encoderBiases = new ArrayList<>();
        List<Matrix> decoderWeights = new ArrayList<>();
        List<Matrix> decoderBiases = new ArrayList<>();

        int imageSize;
        int latentSize;
        double learningRate;

        public FaceGenerator(int imageSize, int latentSize, double learningRate) {
            this.imageSize = imageSize;
            this.latentSize = latentSize;
            this.learningRate = learningRate;

            // Encoder layers
            encoderWeights.add(new Matrix(512, imageSize));
            encoderWeights.get(0).randomize(-0.1, 0.1);
            encoderBiases.add(new Matrix(512, 1));
            encoderBiases.get(0).randomize(-0.1, 0.1);

            encoderWeights.add(new Matrix(256, 512));
            encoderWeights.get(1).randomize(-0.1, 0.1);
            encoderBiases.add(new Matrix(256, 1));
            encoderBiases.get(1).randomize(-0.1, 0.1);

            encoderWeights.add(new Matrix(latentSize * 2, 256));
            encoderWeights.get(2).randomize(-0.1, 0.1);
            encoderBiases.add(new Matrix(latentSize * 2, 1));
            encoderBiases.get(2).randomize(-0.1, 0.1);

            // Decoder layers
            decoderWeights.add(new Matrix(256, latentSize));
            decoderWeights.get(0).randomize(-0.1, 0.1);
            decoderBiases.add(new Matrix(256, 1));
            decoderBiases.get(0).randomize(-0.1, 0.1);

            decoderWeights.add(new Matrix(512, 256));
            decoderWeights.get(1).randomize(-0.1, 0.1);
            decoderBiases.add(new Matrix(512, 1));
            decoderBiases.get(1).randomize(-0.1, 0.1);

            decoderWeights.add(new Matrix(imageSize, 512));
            decoderWeights.get(2).randomize(-0.1, 0.1);
            decoderBiases.add(new Matrix(imageSize, 1));
            decoderBiases.get(2).randomize(-0.1, 0.1);
        }

        public Matrix encode(Matrix input) {
            Matrix h = input.copy();
            LeakyReLU leaky = new LeakyReLU();

            // Encoder forward pass
            h = Matrix.multiply(encoderWeights.get(0), h);
            h = Matrix.add(h, encoderBiases.get(0));
            h.applyFunction(leaky);

            h = Matrix.multiply(encoderWeights.get(1), h);
            h = Matrix.add(h, encoderBiases.get(1));
            h.applyFunction(leaky);

            h = Matrix.multiply(encoderWeights.get(2), h);
            h = Matrix.add(h, encoderBiases.get(2));

            return h; // Returns mean and logvar concatenated
        }

        public Matrix decode(Matrix z) {
            Matrix h = z.copy();
            LeakyReLU leaky = new LeakyReLU();
            Sigmoid sigmoid = new Sigmoid();

            // Decoder forward pass
            h = Matrix.multiply(decoderWeights.get(0), h);
            h = Matrix.add(h, decoderBiases.get(0));
            h.applyFunction(leaky);

            h = Matrix.multiply(decoderWeights.get(1), h);
            h = Matrix.add(h, decoderBiases.get(1));
            h.applyFunction(leaky);

            h = Matrix.multiply(decoderWeights.get(2), h);
            h = Matrix.add(h, decoderBiases.get(2));
            h.applyFunction(sigmoid);

            return h;
        }

        public Matrix sample(Matrix meanLogVar) {
            Matrix mean = new Matrix(latentSize, 1);
            Matrix logVar = new Matrix(latentSize, 1);

            for (int i = 0; i < latentSize; i++) {
                mean.data[i][0] = meanLogVar.data[i][0];
                logVar.data[i][0] = meanLogVar.data[i + latentSize][0];
            }

            Matrix epsilon = new Matrix(latentSize, 1);
            epsilon.randomize(0, 1);

            Matrix z = new Matrix(latentSize, 1);
            for (int i = 0; i < latentSize; i++) {
                z.data[i][0] = mean.data[i][0] + Math.exp(logVar.data[i][0] / 2) * epsilon.data[i][0];
            }

            return z;
        }

        public void train(List<Matrix> dataset, int epochs) {
            LeakyReLU leaky = new LeakyReLU();
            Sigmoid sigmoid = new Sigmoid();

            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalLoss = 0;
                double totalReconLoss = 0;
                double totalKLLoss = 0;

                for (Matrix input : dataset) {
                    // Forward pass - ENCODER
                    Matrix[] encoderActivations = new Matrix[3];
                    Matrix[] encoderPreActivations = new Matrix[3];

                    // Layer 1
                    Matrix h1 = Matrix.multiply(encoderWeights.get(0), input);
                    h1 = Matrix.add(h1, encoderBiases.get(0));
                    encoderPreActivations[0] = h1.copy();
                    h1.applyFunction(leaky);
                    encoderActivations[0] = h1.copy();

                    // Layer 2
                    Matrix h2 = Matrix.multiply(encoderWeights.get(1), h1);
                    h2 = Matrix.add(h2, encoderBiases.get(1));
                    encoderPreActivations[1] = h2.copy();
                    h2.applyFunction(leaky);
                    encoderActivations[1] = h2.copy();

                    // Output layer (mean and logvar)
                    Matrix meanLogVar = Matrix.multiply(encoderWeights.get(2), h2);
                    meanLogVar = Matrix.add(meanLogVar, encoderBiases.get(2));
                    encoderPreActivations[2] = meanLogVar.copy();

                    // Sampling
                    Matrix z = sample(meanLogVar);

                    // Forward pass - DECODER
                    Matrix[] decoderActivations = new Matrix[3];
                    Matrix[] decoderPreActivations = new Matrix[3];

                    // Layer 1
                    Matrix d1 = Matrix.multiply(decoderWeights.get(0), z);
                    d1 = Matrix.add(d1, decoderBiases.get(0));
                    decoderPreActivations[0] = d1.copy();
                    d1.applyFunction(leaky);
                    decoderActivations[0] = d1.copy();

                    // Layer 2
                    Matrix d2 = Matrix.multiply(decoderWeights.get(1), d1);
                    d2 = Matrix.add(d2, decoderBiases.get(1));
                    decoderPreActivations[1] = d2.copy();
                    d2.applyFunction(leaky);
                    decoderActivations[1] = d2.copy();

                    // Output layer
                    Matrix output = Matrix.multiply(decoderWeights.get(2), d2);
                    output = Matrix.add(output, decoderBiases.get(2));
                    decoderPreActivations[2] = output.copy();
                    output.applyFunction(sigmoid);
                    decoderActivations[2] = output.copy();

                    // Calculate losses
                    double reconLoss = 0;
                    Matrix reconError = new Matrix(input.rows, 1);
                    for (int i = 0; i < input.rows; i++) {
                        double diff = input.data[i][0] - output.data[i][0];
                        reconLoss += diff * diff;
                        reconError.data[i][0] = diff;
                    }
                    reconLoss *= 0.5;

                    double klLoss = 0;
                    Matrix klError = new Matrix(latentSize * 2, 1);
                    for (int i = 0; i < latentSize; i++) {
                        double mean = meanLogVar.data[i][0];
                        double logVar = meanLogVar.data[i + latentSize][0];
                        klLoss += -0.5 * (1 + logVar - mean * mean - Math.exp(logVar));

                        // KL error derivatives
                        klError.data[i][0] = mean; // dKL/dmean = mean
                        klError.data[i + latentSize][0] = (Math.exp(logVar) - 1) * 0.5; // dKL/dlogvar = (exp(logvar) -
                                                                                        // 1)/2
                    }

                    totalLoss += reconLoss + klLoss;
                    totalReconLoss += reconLoss;
                    totalKLLoss += klLoss;

                    // BACKPROPAGATION - DECODER
                    Matrix delta = Matrix.scalarMultiply(reconError, -1); // -∂L/∂output

                    // Output layer gradient
                    Matrix outputGrad = Matrix.hadamard(delta, sigmoidDerivative(decoderPreActivations[2]));

                    // Decoder weights update
                    Matrix dW2 = Matrix.multiply(outputGrad, Matrix.transpose(decoderActivations[1]));
                    decoderWeights.set(2,
                            Matrix.subtract(decoderWeights.get(2), Matrix.scalarMultiply(dW2, learningRate)));
                    decoderBiases.set(2,
                            Matrix.subtract(decoderBiases.get(2), Matrix.scalarMultiply(outputGrad, learningRate)));

                    // Backprop through decoder
                    delta = Matrix.multiply(Matrix.transpose(decoderWeights.get(2)), outputGrad);

                    // Hidden layer 2 gradient
                    Matrix h2Grad = Matrix.hadamard(delta, leakyDerivative(decoderPreActivations[1]));
                    Matrix dW1 = Matrix.multiply(h2Grad, Matrix.transpose(decoderActivations[0]));
                    decoderWeights.set(1,
                            Matrix.subtract(decoderWeights.get(1), Matrix.scalarMultiply(dW1, learningRate)));
                    decoderBiases.set(1,
                            Matrix.subtract(decoderBiases.get(1), Matrix.scalarMultiply(h2Grad, learningRate)));

                    delta = Matrix.multiply(Matrix.transpose(decoderWeights.get(1)), h2Grad);

                    // Hidden layer 1 gradient
                    Matrix h1Grad = Matrix.hadamard(delta, leakyDerivative(decoderPreActivations[0]));
                    Matrix dW0 = Matrix.multiply(h1Grad, Matrix.transpose(z));
                    decoderWeights.set(0,
                            Matrix.subtract(decoderWeights.get(0), Matrix.scalarMultiply(dW0, learningRate)));
                    decoderBiases.set(0,
                            Matrix.subtract(decoderBiases.get(0), Matrix.scalarMultiply(h1Grad, learningRate)));

                    // BACKPROPAGATION - ENCODER
                    // Gradient through sampling (reparameterization trick)
                    Matrix zGrad = Matrix.multiply(Matrix.transpose(decoderWeights.get(0)), h1Grad);

                    // KL divergence gradient
                    Matrix klGrad = klError.copy();

                    // Combine gradients
                    Matrix meanLogVarGrad = Matrix.add(zGrad, klGrad);

                    // Output layer gradient
                    Matrix eW2Grad = meanLogVarGrad.copy();
                    Matrix eW2Update = Matrix.multiply(eW2Grad, Matrix.transpose(encoderActivations[1]));
                    encoderWeights.set(2,
                            Matrix.subtract(encoderWeights.get(2), Matrix.scalarMultiply(eW2Update, learningRate)));
                    encoderBiases.set(2,
                            Matrix.subtract(encoderBiases.get(2), Matrix.scalarMultiply(eW2Grad, learningRate)));

                    // Backprop through encoder
                    delta = Matrix.multiply(Matrix.transpose(encoderWeights.get(2)), meanLogVarGrad);

                    // Hidden layer 2 gradient
                    Matrix eh2Grad = Matrix.hadamard(delta, leakyDerivative(encoderPreActivations[1]));
                    Matrix eW1Update = Matrix.multiply(eh2Grad, Matrix.transpose(encoderActivations[0]));
                    encoderWeights.set(1,
                            Matrix.subtract(encoderWeights.get(1), Matrix.scalarMultiply(eW1Update, learningRate)));
                    encoderBiases.set(1,
                            Matrix.subtract(encoderBiases.get(1), Matrix.scalarMultiply(eh2Grad, learningRate)));

                    delta = Matrix.multiply(Matrix.transpose(encoderWeights.get(1)), eh2Grad);

                    // Hidden layer 1 gradient
                    Matrix eh1Grad = Matrix.hadamard(delta, leakyDerivative(encoderPreActivations[0]));
                    Matrix eW0Update = Matrix.multiply(eh1Grad, Matrix.transpose(input));
                    encoderWeights.set(0,
                            Matrix.subtract(encoderWeights.get(0), Matrix.scalarMultiply(eW0Update, learningRate)));
                    encoderBiases.set(0,
                            Matrix.subtract(encoderBiases.get(0), Matrix.scalarMultiply(eh1Grad, learningRate)));
                }

                // Learning rate decay
                if (epoch % 10 == 0) {
                    learningRate *= 0.95;
                }

                System.out.printf("Epoch %d - Loss: %.4f (Recon: %.4f, KL: %.4f) LR: %.6f\n",
                        epoch,
                        totalLoss / dataset.size(),
                        totalReconLoss / dataset.size(),
                        totalKLLoss / dataset.size(),
                        learningRate);
            }
        }

        private Matrix sigmoidDerivative(Matrix m) {
            Matrix result = new Matrix(m.rows, m.cols);
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) {
                    double s = 1 / (1 + Math.exp(-m.data[i][j]));
                    result.data[i][j] = s * (1 - s);
                }
            }
            return result;
        }

        private Matrix leakyDerivative(Matrix m) {
            Matrix result = new Matrix(m.rows, m.cols);
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) {
                    result.data[i][j] = m.data[i][j] > 0 ? 1 : 0.01;
                }
            }
            return result;
        }

    }

    // Image utilities
    public static Matrix imageToMatrix(BufferedImage img, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, width, height, null);
        g.dispose();

        Matrix matrix = new Matrix(width * height * 3, 1);
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = resized.getRGB(x, y);
                matrix.data[index++][0] = ((pixel >> 16) & 0xff) / 255.0;
                matrix.data[index++][0] = ((pixel >> 8) & 0xff) / 255.0;
                matrix.data[index++][0] = (pixel & 0xff) / 255.0;
            }
        }
        return matrix;
    }

    public static BufferedImage matrixToImage(Matrix matrix, int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) (matrix.data[index++][0] * 255);
                int g = (int) (matrix.data[index++][0] * 255);
                int b = (int) (matrix.data[index++][0] * 255);
                int pixel = (r << 16) | (g << 8) | b;
                img.setRGB(x, y, pixel);
            }
        }
        return img;
    }

    public static void main(String[] args) {
        // Configuration - adjust these as needed
        int width = 1280; // Output image width
        int height = 720; // Output image height
        int latentSize = 128; // Size of latent space
        double learningRate = 0.001;
        int epochs = 200;
        String imageDirPath = "your_images_folder"; // Folder containing your face images

        try {
            // 1. Load all face images from directory
            System.out.println("Loading training images from: " + new File(imageDirPath).getAbsolutePath());
            List<Matrix> dataset = loadImageDataset(imageDirPath, width, height);

            if (dataset.size() < 2) {
                System.err.println("ERROR: Need at least 2 images to train. Found only " + dataset.size());
                return;
            }

            // 2. Create and train the face generator
            System.out.println("\nInitializing face generator...");
            int imageSize = width * height * 3; // RGB channels
            FaceGenerator generator = new FaceGenerator(imageSize, latentSize, learningRate);

            System.out.println("Starting training with " + dataset.size() + " images...");
            generator.train(dataset, epochs);

            // 3. Generate new faces
            System.out.println("\nGenerating new faces...");
            for (int i = 0; i < 5; i++) {
                Matrix randomZ = new Matrix(latentSize, 1);
                randomZ.randomize(-2, 2); // Sample from latent space

                Matrix generated = generator.decode(randomZ);
                BufferedImage output = matrixToImage(generated, width, height);
                String outputPath = "generated_face_" + i + ".jpg";
                ImageIO.write(output, "jpg", new File(outputPath));
                System.out.println("Saved: " + outputPath);
            }

            // 4. Create face morph between first two training images
            System.out.println("\nCreating face morph...");
            Matrix face1 = dataset.get(0);
            Matrix face2 = dataset.get(1);

            Matrix encoded1 = generator.encode(face1);
            Matrix encoded2 = generator.encode(face2);

            Matrix z1 = generator.sample(encoded1);
            Matrix z2 = generator.sample(encoded2);

            for (int i = 0; i <= 10; i++) {
                double alpha = i / 10.0;
                Matrix interpolatedZ = interpolateVectors(z1, z2, alpha);

                Matrix morphed = generator.decode(interpolatedZ);
                BufferedImage morphedImg = matrixToImage(morphed, width, height);
                String morphPath = "morph_" + i + ".jpg";
                ImageIO.write(morphedImg, "jpg", new File(morphPath));
                System.out.println("Saved: " + morphPath);
            }

            System.out.println("\nDone! Check your generated files.");

        } catch (Exception e) {
            System.err.println("\nFATAL ERROR:");
            e.printStackTrace();
            System.err.println("\nTROUBLESHOOTING:");
            System.err.println("1. Make sure you have a 'faces' folder in your project");
            System.err.println("2. Put at least 2 JPG/PNG images of your face in it");
            System.err.println("3. Check all images can be opened normally");
        }
    }

    // Helper method to load all images from a directory
    private static List<Matrix> loadImageDataset(String dirPath, int width, int height) throws Exception {
        List<Matrix> dataset = new ArrayList<>();
        File imageDir = new File(dirPath);

        if (!imageDir.exists()) {
            throw new FileNotFoundException("Directory not found: " + imageDir.getAbsolutePath());
        }

        File[] imageFiles = imageDir.listFiles((dir, name) -> name.toLowerCase().matches(".*\\.(jpg|jpeg|png|bmp)$"));

        if (imageFiles == null || imageFiles.length == 0) {
            throw new FileNotFoundException("No images found in: " + dirPath);
        }

        System.out.println("Found " + imageFiles.length + " images");

        for (File imageFile : imageFiles) {
            try {
                BufferedImage img = ImageIO.read(imageFile);
                if (img == null) {
                    System.err.println("Skipping unreadable file: " + imageFile.getName());
                    continue;
                }

                System.out.println("- Processing: " + imageFile.getName() +
                        " (" + img.getWidth() + "x" + img.getHeight() + ")");

                dataset.add(imageToMatrix(img, width, height));
            } catch (Exception e) {
                System.err.println("Error loading " + imageFile.getName() + ": " + e.getMessage());
            }
        }

        return dataset;
    }

    // Helper method for vector interpolation
    private static Matrix interpolateVectors(Matrix a, Matrix b, double alpha) {
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = (1 - alpha) * a.data[i][j] + alpha * b.data[i][j];
            }
        }
        return result;
    }
}