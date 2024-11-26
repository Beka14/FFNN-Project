package layers;

import java.util.Arrays;
import java.util.Random;

public class OutputLayer {
    private float[][] weights;
    private float[] biases;
    public int inputLen;
    public int outputLen;
    private float learningRate;

    private float[] z;  // Raw output before softmax

    private float[] input;
    private float[] softmaxOutput;  // Softmax output

    //need to init these for batch gradient updates - gradient of weight and bias (L_z == L_b)
    float[][] L_w;
    float[] L_softmax;
    float momentum;
    private float[][] velocityWeights;
    private float[] velocityBiases;

    public OutputLayer(int inputLen, int outputLen, float learningRate, float momentum) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;
        this.learningRate = learningRate;
        this.momentum = momentum;

        weights = new float[inputLen][outputLen];
        biases = new float[outputLen];
        velocityWeights = new float[inputLen][outputLen];
        velocityBiases = new float[outputLen];
        L_w = new float[inputLen][outputLen];
        L_softmax = new float[outputLen];
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

    private float[] softMax(float[] pre_activation) {
        float maxLogit = findMax(pre_activation);
        float sum = 0;
        softmaxOutput = new float[pre_activation.length];

        for (int i = 0; i < pre_activation.length; i++) {
            softmaxOutput[i] = (float) Math.exp(pre_activation[i] - maxLogit);
            sum += softmaxOutput[i];
        }

        for (int i = 0; i < pre_activation.length; i++) {
            softmaxOutput[i] /= sum + 1e-7;
        }

        //System.out.println("output layer forward pass after softmax: " + Arrays.toString(softmaxOutput));

        return softmaxOutput;
    }

    public float crossEntropyLoss(float[] predicted, float[] target) {
        float loss = 0;
        for (int j = 0; j < outputLen; j++) {
            loss -= target[j] * Math.log(predicted[j] + 1e-7);
        }
        return loss;
    }

    private float findMax(float[] array) {
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
    }

    public float[] forwardPass(float[] input) {
        this.input = input;
        z = new float[outputLen];

        for (int j = 0; j < outputLen; j++) {
            z[j] = biases[j];
            for (int i = 0; i < inputLen; i++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        //System.out.println("output layer forward pass without softmax: " + Arrays.toString(z));

        return softMax(z);
    }

    public float[] backProp(float[] target) {
        //float[] L_softmax = new float[softmaxOutput.length]; //gradient of loss w.r.t softmax output
        L_softmax = new float[softmaxOutput.length]; //gradient of loss w.r.t softmax output
        for(int i=0; i<softmaxOutput.length; i++){
            L_softmax[i] = softmaxOutput[i] - target[i];
        }

        //gradient of loss w.r.t pre activation z ==> L_z == L_softmax ==> simplification of softmax + cross entropy
        //gradient of loss w.r.t bieses == L_softmax == L_z

        //float[][] L_w = new float[inputLen][outputLen]; //gradient of loss w.r.t weights
        L_w = new float[inputLen][outputLen]; //gradient of loss w.r.t weights
        for(int i=0; i<inputLen; i++){
            for(int j=0; j<outputLen; j++){
                L_w[i][j] = L_softmax[j] * input[i];
                // --- UPDATE WEIGHTS ---
                //weights[i][j] -= learningRate * L_w[i][j];
            }
        }
        // --- UPDATE BIASES ---
        //for(int j=0; j<outputLen; j++){
        //    biases[j] -= learningRate * L_softmax[j];
        //}

        return L_softmax;
    }

    public void updateParameters(float[][] accumulatedGradientsW, float[] accumulatedGradientsB, int batchSize) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= (learningRate / batchSize) * accumulatedGradientsW[i][j];
            }
        }
        for (int j = 0; j < biases.length; j++) {
            biases[j] -= (learningRate / batchSize) * accumulatedGradientsB[j];
        }
    }

    public void updateParametersVelocity(float[][] accumulatedGradientsW, float[] accumulatedGradientsB, int batchSize) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                velocityWeights[i][j] = momentum * velocityWeights[i][j] - (learningRate / batchSize) * accumulatedGradientsW[i][j];
                weights[i][j] += velocityWeights[i][j];
            }
        }
        for (int j = 0; j < biases.length; j++) {
            velocityBiases[j] = momentum * velocityBiases[j] - (learningRate / batchSize) * accumulatedGradientsB[j];
            biases[j] += velocityBiases[j];
        }
    }

    public float[][] getWeightGradients(){
        return L_w;
    }

    public float[] getBiasGradients(){
        return L_softmax;
    }

    public void setLearningRate(float newRate){
        learningRate = newRate;
    }
}
