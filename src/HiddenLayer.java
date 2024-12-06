import java.util.Random;

public class HiddenLayer {
    private float[][] weights;

    private float[] biases;
    public int inputLen;
    public int outputLen;
    private float learningRate;
    private float[][] zBatch;
    private float[][] xBatch; //inputBatch

    //need to init these for batch gradient updates - gradient of weight and bias (L_z == L_b)
    float[][] L_w;
    float[] L_z;

    float momentum;

    private float[][] gradientsW;
    private float[] gradientsB;
    public HiddenLayer(int inputLen, int outputLen, float learningRate, float momentum) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;

        this.learningRate = learningRate;
        this.momentum = momentum;

        biases = new float[outputLen];
        weights = new float[inputLen][outputLen];

        L_w = new float[inputLen][outputLen];
        L_z = new float[outputLen];
        setWeights();
    }

    public float[][] forwardPass(float[][] inputBatch){
        xBatch = inputBatch;

        // Z = X * W +B
        zBatch = MatrixOperations.matrixMultiply(inputBatch, weights);
        addBias(zBatch, biases);

        return activationFunction(zBatch);
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

    public float[][] activationFunction(float[][] zBatch) {
        int rows = zBatch.length;
        int cols = zBatch[0].length;
        float[][] activatedBatch = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                activatedBatch[i][j] = ActivationFunctions.LeakyReLU(zBatch[i][j]);
            }
        }

        return activatedBatch;
    }

    public float[][] activationDFunction(float[][] zBatch) {
        int rows = zBatch.length;
        int cols = zBatch[0].length;
        float[][] activatedBatch = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                activatedBatch[i][j] = Derivatives.D_LeakyReLU(zBatch[i][j]);
            }
        }

        return activatedBatch;
    }

    public float[][] backProp(float[][] L_y) {
        //L_y == gradient of loss w.r.t activations from the next layer
        //compute gradients of loss L with respect to w, b, inputs x:
        //gradient of loss with respect to pre-activated values z
        //float[] L_z = new float[outputLen]; // gradient of loss with respect to pre-activation values
        float[][] activationD = activationDFunction(zBatch);
        float[][] gradientsZ = MatrixOperations.elementWiseMultiply(L_y, activationD);

        gradientsW = MatrixOperations.matrixMultiply(MatrixOperations.transpose(xBatch), gradientsZ);
        gradientsB = MatrixOperations.sumAlongAxis(gradientsZ, 0);

        return MatrixOperations.matrixMultiply(gradientsZ, MatrixOperations.transpose(weights));
    }

    public void updateParameters() {
        // Update weights: W = W - learningRate * gradientsW
        weights = MatrixOperations.subtractMatrix(weights, MatrixOperations.scalarMultiply(gradientsW, learningRate / xBatch.length));

        // Update biases: B = B - learningRate * gradientsB
        biases = MatrixOperations.subtractVector(biases, MatrixOperations.scalarMultiply(gradientsB, learningRate / xBatch.length));
    }

    public void setWeights(){
        Random r = new Random();

        float stddev = (float) Math.sqrt(2.0 / inputLen);
        for (int i = 0; i < inputLen; i++) {
            for (int j = 0; j < outputLen; j++) {
                weights[i][j] = (float) r.nextGaussian() * stddev;
            }
        }

        for (int j = 0; j < outputLen; j++) {
            biases[j] = 0f;
        }
    }

    public float[][] getWeightGradients(){
        return L_w;
    }

    public float[] getBiasGradients(){
        return L_z;
    }

    public void setLearningRate(float newRate){
        learningRate = newRate;
    }
}
