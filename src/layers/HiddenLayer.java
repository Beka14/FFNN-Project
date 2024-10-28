package layers;

import math.ActivationFunctions;
import math.Derivatives;

import java.util.Random;

public class HiddenLayer extends Layer {
    private float[][] weights;
    private int inputLen;
    private int outputLen;
    private float learningRate;
    private float[] z;
    private float[] x;
    public HiddenLayer(int inputLen, int outputLen) {
        this.inputLen = inputLen;
        this.outputLen = outputLen;

        this.x = new float[inputLen];

        learningRate = 0.01F;

        weights = new float[inputLen][outputLen];
        setWeights();
    }

    public float[] forwardPass(float[] input){
        float[] out = new float[outputLen];
        x = input;
        z = new float[outputLen];

        for(int i=0; i<inputLen; i++){
            for(int j=0; j<outputLen; j++){
                z[j] += input[i] * weights[i][j];
            }
        }

        for(int j = 0; j < outputLen; j++){
            out[j] = ActivationFunctions.ReLU(z[j]);
        }

        return out;
    }
    @Override
    public float[] calcOutput(float[] data) {
        float[] fp = forwardPass(data);
        return (nextLayer != null) ? nextLayer.calcOutput(fp) : fp;
    }

    @Override
    public void backProp(float[] dloss_doutput) {
        float[] dLdX = new float[inputLen];

        for(int k=0; k<inputLen; k++){ //for each input neuron k, calculate the gradint of loss with respect to EACH w[k][j]
            float sum_dLdX = 0;

            for(int j=0; j<outputLen; j++){ //for each output neuron ...
                float dOdZ = Derivatives.D_ReLU(z[j]); //der of relu act.func. with respect to z[j]
                float dZdW = x[k]; //der. of z[j] with respect to w[k][j] => this is simply the input x[k]

                // GRADIENT CALCULATION
                float dLdW = dloss_doutput[j] * dOdZ * dZdW; //gradient of loss with respect to w[k][j]

                //then adjust the w[k][j] by SUBTRACTING the scaled gradient

                weights[k][j] -= dLdW * learningRate;

                //for each input neuron k we accumulate the gradient loss
                sum_dLdX += dloss_doutput[j] * dOdZ * weights[k][j];
            }

            dLdX[k] = sum_dLdX;
        }

        //then we pass the gradient of loss with respect to layers inputs to the PREV LAYER => backprop

        if(prevLayer != null){
            prevLayer.backProp(dLdX);
        }
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

}
