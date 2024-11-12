import data.DataManager;
import layers.HiddenLayer;
import layers.OutputLayer;

import java.io.IOException;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Arrays;

import static data.DataManager.getMiniBatches;

public class FeedFrwdNeuralNetwork {
    private HiddenLayer[] hiddenLayers;
    private OutputLayer outputLayer;
    private float learningRate;
    private float momentum;

    public FeedFrwdNeuralNetwork(int inputLen, int[] hiddenLayerSizes, int outputLen, float learningRate, float momentum) {

        hiddenLayers = new HiddenLayer[hiddenLayerSizes.length];

        hiddenLayers[0] = new HiddenLayer(inputLen, hiddenLayerSizes[0], learningRate, momentum);
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            hiddenLayers[i] = new HiddenLayer(hiddenLayerSizes[i - 1], hiddenLayerSizes[i], learningRate, momentum); // Connect each layer to the previous
        }

        outputLayer = new OutputLayer(hiddenLayerSizes[hiddenLayerSizes.length - 1], outputLen, learningRate, momentum);

        this.learningRate = learningRate;
    }

    public float[] forwardPass(float[] input) {
        float[] hiddenOutput = input;
        for (HiddenLayer layer : hiddenLayers) {
            hiddenOutput = layer.forwardPass(hiddenOutput);
        }
        return outputLayer.forwardPass(hiddenOutput);
    }

    public float calculateLoss(float[] predicted, float[] target) {
        return outputLayer.crossEntropyLoss(predicted, target);
    }

    public void backPropagate(float[] target) {

        float[] nextLayerError = outputLayer.backProp(target);

        for (int i = hiddenLayers.length - 1; i >= 0; i--) {
            nextLayerError = hiddenLayers[i].backProp(nextLayerError);
        }
    }

    public void train(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for(int epoch=0; epoch < epochs; epoch++){
            float totalLoss = 0;
            for(int i=0; i < trainData.length; i++){

                //forward pass
                float[] predictions = forwardPass(trainData[i]);

                //loss computation
                float[] target = new float[10];
                target[trainLabels[i]] = 1.0f;
                //System.out.println("label: " + trainLabels[i] + " target: " + Arrays.toString(target));
                float loss = calculateLoss(predictions, target);
                totalLoss += loss;

                //backprop
                backPropagate(target);

                //update weights here if using batch updates
            }
            System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / trainData.length));
        }
    }

    public void trainBatch(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {

            // Create mini-batches using the MiniBatch class
            DataManager.MiniBatch[] miniBatches = getMiniBatches(trainData, trainLabels, batchSize);

            float totalLoss = 0;

            for (DataManager.MiniBatch batch : miniBatches) {
                // Extract data and labels for the current batch
                float[][] batchData = batch.data;
                float[][] batchLabels = new float[batch.labels.length][10];
                for (int i = 0; i < batch.labels.length; i++) {
                    batchLabels[i][(int) batch.labels[i]] = 1.0f; // Convert labels to one-hot encoding
                }

                // Initialize accumulators for gradients
                float[][] accumulatedGradientsW1 = new float[hiddenLayers[0].inputLen][hiddenLayers[0].outputLen];
                float[] accumulatedGradientsB1 = new float[hiddenLayers[0].outputLen];
                float[][] accumulatedGradientsW2 = new float[hiddenLayers[1].inputLen][hiddenLayers[1].outputLen];
                float[] accumulatedGradientsB2 = new float[hiddenLayers[1].outputLen];
                float[][] accumulatedGradientsW3 = new float[outputLayer.inputLen][outputLayer.outputLen];
                float[] accumulatedGradientsB3 = new float[outputLayer.outputLen];

                for (int i = 0; i < batchData.length; i++) {
                    // Forward pass
                    float[] hiddenOutput1 = hiddenLayers[0].forwardPass(batchData[i]);
                    float[] hiddenOutput2 = hiddenLayers[1].forwardPass(hiddenOutput1);
                    float[] predictions = outputLayer.forwardPass(hiddenOutput2);

                    // Compute loss
                    float loss = outputLayer.crossEntropyLoss(predictions, batchLabels[i]);
                    totalLoss += loss;

                    // Backpropagation
                    float[] gradientOutput = outputLayer.backProp(batchLabels[i]);
                    float[] gradientHidden2 = hiddenLayers[1].backProp(gradientOutput);
                    float[] gradientHidden1 = hiddenLayers[0].backProp(gradientHidden2);

                    // Accumulate gradients for weights and biases
                    accumulateGradients(accumulatedGradientsW1, hiddenLayers[0].getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB1, hiddenLayers[0].getBiasGradients());
                    accumulateGradients(accumulatedGradientsW2, hiddenLayers[1].getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB2, hiddenLayers[1].getBiasGradients());
                    accumulateGradients(accumulatedGradientsW3, outputLayer.getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB3, outputLayer.getBiasGradients());
                }

                // Update parameters for each layer
                hiddenLayers[0].updateParametersVelocity(accumulatedGradientsW1, accumulatedGradientsB1, batchData.length);
                hiddenLayers[1].updateParametersVelocity(accumulatedGradientsW2, accumulatedGradientsB2, batchData.length);
                outputLayer.updateParametersVelocity(accumulatedGradientsW3, accumulatedGradientsB3, batchData.length);
            }

            // Log the epoch loss
            System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / trainData.length));
        }
    }


    private void accumulateGradients(float[][] accumulated, float[][] gradients) {
        for (int i = 0; i < accumulated.length; i++) {
            for (int j = 0; j < accumulated[i].length; j++) {
                accumulated[i][j] += gradients[i][j];
            }
        }
    }

    private void accumulateBiasGradients(float[] accumulated, float[] gradients) {
        for (int i = 0; i < accumulated.length; i++) {
            accumulated[i] += gradients[i];
        }
    }

    public void validate(float[][] valData, int[] valLabels){
        float valLoss = 0;
        int correct = 0;

        System.out.println("=== BEGIN VALIDATION ===");

        for (int i = 0; i < valData.length; i++) {
            float[] hiddenOutput1 = hiddenLayers[0].forwardPass(valData[i]);
            float[] hiddenOutput2 = hiddenLayers[1].forwardPass(hiddenOutput1);
            float[] predictions = outputLayer.forwardPass(hiddenOutput2);

            float[] target = new float[10];
            target[valLabels[i]] = 1.0f;
            // Compute validation loss
            valLoss += outputLayer.crossEntropyLoss(predictions, target);

            // Compute validation accuracy
            int predictedClass = getArgMax(predictions);
            int actualClass = getArgMax(target);
            if (predictedClass == actualClass) {
                correct++;
            }
        }

        float valAccuracy = (float) correct / valData.length;
        System.out.printf("Validation Loss: %.4f, Validation Accuracy: %.2f%%\n", valLoss / valData.length, valAccuracy * 100);
    }

    private int getArgMax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void main(String[] args) throws IOException {
        int inputLen = 784; 
        int[] hiddenLayerSizes = {256, 128};
        int outputLen = 10;
        float learningRate = 0.01f;
        float momentum = 0.9f;
        int batchSize = 128;
        int epochs = 20;

        //68% lr = 0.01f; m = 0.5
        //69%
        //70% learningRate = 0.008f; momentum = 0.7f;

        FeedFrwdNeuralNetwork neuralNetwork = new FeedFrwdNeuralNetwork(inputLen, hiddenLayerSizes, outputLen, learningRate, momentum);

        float[][] trainData;
        int[] trainLabels;

        float[][] ValidationData;
        int[] ValidationLabels;

        trainData = DataManager.loadImageData("data/fashion_mnist_train_vectors.csv", 45000, 784);
        trainLabels = DataManager.loadLabels("data/fashion_mnist_train_labels.csv", 45000); //60000

        ValidationData = DataManager.loadImageData("data/fashion_mnist_test_vectors.csv", 12000, 784);
        ValidationLabels = DataManager.loadLabels("data/fashion_mnist_test_labels.csv", 12000);

        //System.out.println(Arrays.deepToString(trainData));
        //System.out.println(Arrays.toString(trainLabels));

        //neuralNetwork.train(trainData, trainLabels, batchSize, epochs);
        System.out.println("start:" + LocalTime.now());
        neuralNetwork.trainBatch(trainData, trainLabels, batchSize, epochs);
        neuralNetwork.validate(ValidationData, ValidationLabels);
        System.out.println("end:" + LocalTime.now());
    }
}
