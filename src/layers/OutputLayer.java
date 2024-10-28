package layers;

import java.util.Arrays;
import java.util.Random;

public class OutputLayer {
    private float[][] weights;
    private int inputLen;
    private int outputLen;
    private float learningRate;

    private float[] z;  // Raw output before softmax
    private float[] softmaxOutput;  // Softmax output

    public OutputLayer(int inputLen, int outputLen, float learningRate) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;
        this.learningRate = learningRate;

        weights = new float[inputLen][outputLen];
        setWeights();
    }

    public void setWeights(){
        Random r = new Random();

        float stddev = (float) Math.sqrt(2.0 / inputLen);
        for (int i = 0; i < outputLen; i++) {
            for (int j = 0; j < inputLen; j++) {
                weights[i][j] = (float) r.nextGaussian() * stddev;
            }
        }
    }

    private float[] softMax(float[] logits) {
        //float maxLogit = findMax(logits);
        float sum = 0;
        softmaxOutput = new float[logits.length];

        for (int i = 0; i < logits.length; i++) {
            //softmaxOutput[i] = (float) Math.exp(logits[i] - maxLogit);
            softmaxOutput[i] = (float) Math.exp(logits[i]);
            sum += softmaxOutput[i];
        }

        for (int i = 0; i < logits.length; i++) {
            softmaxOutput[i] /= sum;
        }

        return softmaxOutput;
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
        z = new float[outputLen];
        for (int i = 0; i < inputLen; i++) {
            for (int j = 0; j < outputLen; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        return softMax(z);
    }

    public float crossEntropyLoss(float[] target) {
        float loss = 0;
        for (int i = 0; i < outputLen; i++) {
            loss -= target[i] * Math.log(softmaxOutput[i]);
        }
        return loss;
    }

    public void backProp(float[] target) {
        //with softmax and CE combined the backprop is simplified due to the gradient of CE loss
        //w.r.t logits z becomes softmaxOutput[j] - target[j]

        float[] dLoss_dZ = new float[outputLen];  // Gradient of loss w.r.t. z

        for (int j = 0; j < outputLen; j++) {
            dLoss_dZ[j] = softmaxOutput[j] - target[j];  // Gradient of combined softmax and cross-entropy
        }

        // Gradient descent update for weights
        for (int i = 0; i < inputLen; i++) {
            for (int j = 0; j < outputLen; j++) {
                weights[i][j] -= learningRate * dLoss_dZ[j] * z[i];
            }
        }

        // No need to backpropagate further since this is the output layer
    }
}
