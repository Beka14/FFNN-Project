import data.DataManager;
import data.DataManager.MiniBatch;
import layers.HiddenLayer;
import layers.OutputLayer;

public class FeedFrwdNeuralNetwork {
    private HiddenLayer[] hiddenLayers;
    private OutputLayer outputLayer;
    private float learningRate;

    public FeedFrwdNeuralNetwork(int inputLen, int hiddenLen, int outputLen,int numHidden, float learningRate) {
        hiddenLayers = new HiddenLayer[numHidden];
        for(int i=0; i <numHidden; i++){
            hiddenLayers[i] = new HiddenLayer(inputLen, hiddenLen);
            //connect them by .nex . prevlayer ???
        }
        outputLayer = new OutputLayer(hiddenLen, outputLen, learningRate);
        //connect ???
        this.learningRate = learningRate;
    }

    public float[] forwardPass(float[] input) {
        float[] hiddenOutput = hiddenLayers[0].calcOutput(input); //TODO: ???? for each or first/last??
        return outputLayer.forwardPass(hiddenOutput); //TODO do i need this ??
    }

    public float calculateLoss(float[] predicted, float[] target) {
        return outputLayer.crossEntropyLoss(target);
    }

    public void backPropagate(float[] predicted, float[] target) {
        float[] outputError = outputLayer.forwardPass(predicted);
        outputLayer.backProp(target);
        hiddenLayers[hiddenLayers.length-1].backProp(outputError); //TODO ????
    }

    public void train(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            MiniBatch[] batches = DataManager.getMiniBatches(trainData, trainLabels, batchSize);

            for (MiniBatch batch : batches) {
                float[][] batchData = batch.data;
                float[] batchLabels = batch.labels;

                for (int i = 0; i < batchData.length; i++) {
                    float[] input = batchData[i];
                    float[] target = new float[10];
                    target[(int) batchLabels[i]] = 1.0f;

                    //forward pass
                    float[] predicted = forwardPass(input);

                    //compute loss
                    float loss = calculateLoss(predicted, target);

                    //backpropagation and weight update
                    backPropagate(predicted, target);
                }
            }

            System.out.println("Epoch " + epoch + " completed.");
        }
    }
}
