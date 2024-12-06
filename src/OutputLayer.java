import java.util.Random;

public class OutputLayer {
    private float[][] weights;
    private float[] biases;
    public int inputLen;
    public int outputLen;
    private float learningRate;

    private float[][] velocityWeights;
    private float[] velocityBiases;

    private float[][] zBatch;  // raw output before softmax

    private float[][] xBatch; // inoutBatch
    private float[][] softmaxOutput;  // Softmax output

    //need to init these for batch gradient updates - gradient of weight and bias (L_z == L_b)
    float momentum;
    private float[][] gradientsW;
    private float[] gradientsB;

    public OutputLayer(int inputLen, int outputLen, float learningRate, float momentum) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;
        this.learningRate = learningRate;
        this.momentum = momentum;

        weights = new float[inputLen][outputLen];
        biases = new float[outputLen];

        this.velocityWeights = new float[inputLen][outputLen];
        this.velocityBiases = new float[outputLen];

        setWeights();
    }

    public void setWeights(){
        Random r = new Random();

        float range = (float) Math.sqrt(6.0 / (inputLen + outputLen));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = (float) (r.nextDouble() * 2 * range - range);
            }
        }
        for (int j = 0; j < outputLen; j++) {
            biases[j] = 0.0f;
        }
    }

    private double[] clipGradients(double[] gradients, double threshold) {
        double norm = 0.0;
        for (double grad : gradients) {
            norm += grad * grad;
        }
        norm = Math.sqrt(norm);

        if (norm > threshold) {
            double scalingFactor = threshold / norm;
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scalingFactor;
            }
        }
        return gradients;
    }

    public static float[][] softmax(float[][] zBatch) {
        int rows = zBatch.length;
        int cols = zBatch[0].length;
        float[][] softmaxBatch = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            // find max for dstability
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                maxLogit = Math.max(maxLogit, zBatch[i][j]);
            }

            float sumExp = 0.0f;
            for (int j = 0; j < cols; j++) {
                softmaxBatch[i][j] = (float) Math.exp(zBatch[i][j] - maxLogit); // subtract maxLogit for numerical stability
                sumExp += softmaxBatch[i][j];
            }

            // normalize by the sum of exponentials
            for (int j = 0; j < cols; j++) {
                softmaxBatch[i][j] /= sumExp;
            }
        }

        return softmaxBatch;
    }

    public float crossEntropyLossBatch(float[][] predictions, float[][] targets) {
        float loss = 0;
        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                loss -= targets[i][j] * Math.log(predictions[i][j] + 1e-7); // for numerical stability
            }
        }
        return loss / predictions.length;
    }

    public float[][] forwardPass(float[][] inputBatch) {
        xBatch = inputBatch;

        zBatch = MatrixOperations.matrixMultiply(inputBatch, weights);
        addBias(zBatch, biases);

        softmaxOutput = softmax(zBatch);
        return softmaxOutput;
    }

    public static void addBias(float[][] matrix, float[] biases) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] += biases[j];
            }
        }
    }

    public float[][] backProp(float[][] targets) {
        //gradient of loss w.r.t softmax output
        float[][] gradientsZ = MatrixOperations.subtractMatrix(softmaxOutput, targets);

        //gradient of loss w.r.t pre activation z ==> L_z == L_softmax ==> simplification of softmax + cross entropy
        //gradient of loss w.r.t bieses == L_softmax == L_z

       //gradient of loss w.r.t weights
        gradientsW = MatrixOperations.matrixMultiply(MatrixOperations.transpose(xBatch), gradientsZ);
        gradientsB = MatrixOperations.sumAlongAxis(gradientsZ, 0);

        //gradient of loss w.r.t inputs
        return MatrixOperations.matrixMultiply(gradientsZ, MatrixOperations.transpose(weights));
    }

    public void updateParametersMomentum(int batchSize) {
        velocityWeights = MatrixOperations.addMatrix(
                MatrixOperations.scalarMultiply(velocityWeights, momentum),
                MatrixOperations.scalarMultiply(gradientsW, -(learningRate / batchSize))
        );
        weights = MatrixOperations.addMatrix(weights, velocityWeights);

        // update velocities and biases
        velocityBiases = MatrixOperations.addVector(
                MatrixOperations.scalarMultiply(velocityBiases, momentum),
                MatrixOperations.scalarMultiply(gradientsB, -(learningRate / batchSize))
        );
        biases = MatrixOperations.addVector(biases, velocityBiases);
    }

    public void setLearningRate(float newRate){
        learningRate = newRate;
    }
}
