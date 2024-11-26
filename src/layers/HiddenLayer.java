package layers;

import math.ActivationFunctions;
import math.Derivatives;

import java.util.Arrays;
import java.util.Random;

public class HiddenLayer extends Layer {
    private float[][] weights;

    private float[] biases;
    public int inputLen;
    public int outputLen;
    private float learningRate;
    private float[] z;
    private float[] x;

    private float[] dropout_mask;
    private float dropout_rate;
    private boolean training;

    //need to init these for batch gradient updates - gradient of weight and bias (L_z == L_b)
    float[][] L_w;
    float[] L_z;

    float momentum;

    private float[][] velocityWeights;
    private float[] velocityBiases;
    public HiddenLayer(int inputLen, int outputLen, float learningRate, float momentum, float dropout_rate) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;
        this.dropout_rate = dropout_rate;
        training = true;

        this.x = new float[inputLen];

        this.learningRate = learningRate;
        this.momentum = momentum;

        biases = new float[outputLen];
        weights = new float[inputLen][outputLen];
        velocityWeights = new float[inputLen][outputLen];
        velocityBiases = new float[outputLen];

        L_w = new float[inputLen][outputLen];
        L_z = new float[outputLen];
        setWeights();
    }

    public float[] forwardPass(float[] input){
        float[] out = new float[outputLen];
        x = input;
        z = new float[outputLen];

        for (int j = 0; j < outputLen; j++) {
            z[j] = biases[j];
            for (int i = 0; i < inputLen; i++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        for(int j = 0; j < outputLen; j++){
            out[j] = ActivationFunctions.LeakyReLU(z[j]);
        }


        if (training) {
            dropout_mask = new float[outputLen];
            for (int j = 0; j < outputLen; j++) {
                dropout_mask[j] = Math.random() > dropout_rate ? 1.0f : 0.0f;
                out[j] *= dropout_mask[j];
                out[j] /= (1 - dropout_rate);
            }
        }

        //System.out.println("Hidden layer forward pass: " + Arrays.toString(out));

        return out;
    }

    @Override
    public float[] backProp(float[] L_y) {
        //L_y == gradient of loss w.r.t activations from the next layer
        //compute gradients of loss L with respect to w, b, inputs x:
        //gradient of loss with respect to pre-activated values z
        //float[] L_z = new float[outputLen]; // gradient of loss with respect to pre-activation values
        L_z = new float[outputLen]; // gradient of loss with respect to pre-activation values
        float[] L_x = new float[inputLen]; //gradient of loss w.r.t inputs
        //float[][] L_w = new float[inputLen][outputLen]; //gradient of loss w.r.t weights
        L_w = new float[inputLen][outputLen]; //gradient of loss w.r.t weights

        //recieve gradients from the next layer
        for(int r=0; r<L_y.length; r++){
            for(int j=0; j<outputLen; j++){
                L_z[r] += L_y[r] * weights[r][j];
            }
        }

        for(int j=0; j<outputLen; j++){
            L_z[j] *= Derivatives.D_LeakyReLU(z[j]);
            if (training) {
                L_z[j] *= dropout_mask[j]; // Apply the same dropout mask during backprop
            }
        }

        for(int i=0;i<inputLen;i++){
            for(int j=0;j<outputLen;j++){
                L_w[i][j] = L_z[j] * x[i]; //x = input ti the hidden neuron
                L_x[i] += weights[i][j] * L_z[j];
                // --- WEIGHTS UPDATE ---
                //weights[i][j] -= learningRate * L_w[i][j];
            }
        }

        //gradient of loss w.r.t biases == L_z
        //gradient of loss w.r.t inputs

        //for(int j=0;j<outputLen;j++){
            // --- BIAS UPDATE ---
        //    biases[j] -= learningRate * L_z[j];
        //}

        return L_z;
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
        return L_z;
    }

    public void setLearningRate(float newRate){
        learningRate = newRate;
    }

    public void setTraining(boolean training) { this.training = training; }


}
