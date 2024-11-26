import data.DataManager;
import data.FileUtil;
import layers.HiddenLayer;
import layers.OutputLayer;

import java.io.IOException;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static data.DataManager.*;

public class FeedFrwdNeuralNetwork {
    private HiddenLayer[] hiddenLayers;
    private OutputLayer outputLayer;
    private float learningRate;

    private static float[][] validateData;
    private static int[] validateLabels;
    private float decayRate = 0.0f;
    private float momentum;

    public FeedFrwdNeuralNetwork(int inputLen, int[] hiddenLayerSizes, int outputLen, float learningRate, float momentum, float dropout) {

        hiddenLayers = new HiddenLayer[hiddenLayerSizes.length];

        hiddenLayers[0] = new HiddenLayer(inputLen, hiddenLayerSizes[0], learningRate, momentum, dropout);
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            hiddenLayers[i] = new HiddenLayer(hiddenLayerSizes[i - 1], hiddenLayerSizes[i], learningRate, momentum, dropout); // Connect each layer to the previous
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

    public void trainBatch3(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {

            float decayingLR = (float) (learningRate * Math.exp(-decayRate * (epoch+1)));
            System.out.println("LEARNING RATE: " + decayingLR);

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

                for(HiddenLayer hl : hiddenLayers){
                    hl.setLearningRate(decayingLR);
                }
                outputLayer.setLearningRate(decayingLR);

                // Initialize accumulators for gradients
                float[][] accumulatedGradientsW1 = new float[hiddenLayers[0].inputLen][hiddenLayers[0].outputLen];
                float[] accumulatedGradientsB1 = new float[hiddenLayers[0].outputLen];
                float[][] accumulatedGradientsW2 = new float[hiddenLayers[1].inputLen][hiddenLayers[1].outputLen];
                float[] accumulatedGradientsB2 = new float[hiddenLayers[1].outputLen];
                float[][] accumulatedGradientsW3 = new float[hiddenLayers[2].inputLen][hiddenLayers[2].outputLen];
                float[] accumulatedGradientsB3 = new float[hiddenLayers[2].outputLen];
                float[][] accumulatedGradientsWO = new float[outputLayer.inputLen][outputLayer.outputLen];
                float[] accumulatedGradientsBO = new float[outputLayer.outputLen];

                for (int i = 0; i < batchData.length; i++) {
                    // Forward pass
                    float[] hiddenOutput1 = hiddenLayers[0].forwardPass(batchData[i]);
                    float[] hiddenOutput2 = hiddenLayers[1].forwardPass(hiddenOutput1);
                    float[] hiddenOutput3 = hiddenLayers[2].forwardPass(hiddenOutput2);
                    float[] predictions = outputLayer.forwardPass(hiddenOutput3);

                    // Compute loss
                    float loss = outputLayer.crossEntropyLoss(predictions, batchLabels[i]);
                    totalLoss += loss;

                    // Backpropagation
                    float[] gradientOutput = outputLayer.backProp(batchLabels[i]);
                    float[] gradientHidden3 = hiddenLayers[2].backProp(gradientOutput);
                    float[] gradientHidden2 = hiddenLayers[1].backProp(gradientHidden3);
                    float[] gradientHidden1 = hiddenLayers[0].backProp(gradientHidden2);

                    // Accumulate gradients for weights and biases
                    accumulateGradients(accumulatedGradientsW1, hiddenLayers[0].getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB1, hiddenLayers[0].getBiasGradients());
                    accumulateGradients(accumulatedGradientsW2, hiddenLayers[1].getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB2, hiddenLayers[1].getBiasGradients());
                    accumulateGradients(accumulatedGradientsW3, hiddenLayers[2].getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsB3, hiddenLayers[2].getBiasGradients());
                    accumulateGradients(accumulatedGradientsWO, outputLayer.getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsBO, outputLayer.getBiasGradients());
                }

                // Update parameters for each layer
                hiddenLayers[0].updateParametersVelocity(accumulatedGradientsW1, accumulatedGradientsB1, batchData.length);
                hiddenLayers[1].updateParametersVelocity(accumulatedGradientsW2, accumulatedGradientsB2, batchData.length);
                hiddenLayers[2].updateParametersVelocity(accumulatedGradientsW3, accumulatedGradientsB3, batchData.length);
                outputLayer.updateParametersVelocity(accumulatedGradientsWO, accumulatedGradientsBO, batchData.length);
            }

            // Log the epoch loss
            System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / trainData.length));
        }
    }


    public void trainBatch2(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {

            float decayingLR = (float) (learningRate * Math.exp(-decayRate * epoch));
            System.out.println("LEARNING RATE: " + decayingLR);

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

                for(HiddenLayer hl : hiddenLayers){
                    hl.setLearningRate(decayingLR);
                }
                outputLayer.setLearningRate(decayingLR);

                // Initialize accumulators for gradients
                float[][] accumulatedGradientsW1 = new float[hiddenLayers[0].inputLen][hiddenLayers[0].outputLen];
                float[] accumulatedGradientsB1 = new float[hiddenLayers[0].outputLen];
                float[][] accumulatedGradientsW2 = new float[hiddenLayers[1].inputLen][hiddenLayers[1].outputLen];
                float[] accumulatedGradientsB2 = new float[hiddenLayers[1].outputLen];
                float[][] accumulatedGradientsWO = new float[outputLayer.inputLen][outputLayer.outputLen];
                float[] accumulatedGradientsBO = new float[outputLayer.outputLen];

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
                    accumulateGradients(accumulatedGradientsWO, outputLayer.getWeightGradients());
                    accumulateBiasGradients(accumulatedGradientsBO, outputLayer.getBiasGradients());
                }

                // Update parameters for each layer
                hiddenLayers[0].updateParametersVelocity(accumulatedGradientsW1, accumulatedGradientsB1, batchData.length);
                hiddenLayers[1].updateParametersVelocity(accumulatedGradientsW2, accumulatedGradientsB2, batchData.length);
                outputLayer.updateParametersVelocity(accumulatedGradientsWO, accumulatedGradientsBO, batchData.length);
            }

            // Log the epoch loss
            System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / trainData.length));
            //float[] v = validate2(validateData, validateLabels);
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

    public float[] validate3(float[][] valData, int[] valLabels){
        float valLoss = 0;
        int correct = 0;

        System.out.println("=== BEGIN VALIDATION ===");
        hiddenLayers[0].setTraining(false);
        hiddenLayers[1].setTraining(false);
        hiddenLayers[2].setTraining(false);

        for (int i = 0; i < valData.length; i++) {
            float[] hiddenOutput1 = hiddenLayers[0].forwardPass(valData[i]);
            float[] hiddenOutput2 = hiddenLayers[1].forwardPass(hiddenOutput1);
            float[] hiddenOutput3 = hiddenLayers[2].forwardPass(hiddenOutput2);
            float[] predictions = outputLayer.forwardPass(hiddenOutput3);

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
        return new float[] {valLoss / valData.length, valAccuracy * 100};
    }

    public float[] validate2(float[][] valData, int[] valLabels){
        float valLoss = 0;
        int correct = 0;

        System.out.println("=== BEGIN VALIDATION ===");
        hiddenLayers[0].setTraining(false);
        hiddenLayers[1].setTraining(false);

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
            //System.out.println("prediction: " + predictedClass + " actual: " + actualClass);
            if (predictedClass == actualClass) {
                correct++;
            }
        }

        float valAccuracy = (float) correct / valData.length;
        System.out.printf("Validation Loss: %.4f, Validation Accuracy: %.2f%%\n", valLoss / valData.length, valAccuracy * 100);
        return new float[] {valLoss / valData.length, valAccuracy * 100};
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
        int[] hiddenLayerSizes = {128, 64};
        int outputLen = 10;
        float learningRate = 0.001f;
        float momentum = 0.9f;
        float dropout = 0.0f;
        int batchSize = 32;
        int epochs = 20;

        //68% lr = 0.01f; m = 0.5
        //69%
        //70% learningRate = 0.008f; momentum = 0.7f;

        FeedFrwdNeuralNetwork neuralNetwork = new FeedFrwdNeuralNetwork(inputLen, hiddenLayerSizes, outputLen, learningRate, momentum, dropout);
        neuralNetwork.decayRate = 0.00F;

        float[][] trainData;
        int[] trainLabels;

        DataManager dm = new DataManager();

        float[][] images = dm.loadImageData("data/fashion_mnist_train_vectors.csv", 60000, 784, true);
        int[] labels = dm.loadLabels("data/fashion_mnist_train_labels.csv", 60000);

        //trainData = dm.loadImageData("data/fashion_mnist_train_vectors.csv", 45000, 784, true);
        //trainLabels = dm.loadLabels("data/fashion_mnist_train_labels.csv", 45000); //60000

        //validateData = dm.loadImageData("data/fashion_mnist_test_vectors.csv", 12000, 784, false);
        //validateLabels = dm.loadLabels("data/fashion_mnist_test_labels.csv", 12000);

        DataManager.DataSplit split = splitData(images, labels, 0.8f);
        trainData = split.trainData;
        trainLabels = split.trainLabels;;

        validateData = split.valData;
        validateLabels = split.valLabels;

        //System.out.println(Arrays.deepToString(trainData));
        //System.out.println(Arrays.toString(trainLabels));

        //neuralNetwork.train(trainData, trainLabels, batchSize, epochs);
        System.out.println("start:" + LocalTime.now());
        neuralNetwork.trainBatch2(trainData, trainLabels, batchSize, epochs);

        float[] v = neuralNetwork.validate2(validateData, validateLabels);
        System.out.println("end:" + LocalTime.now());

        float[][] finalValData = dm.loadImageData("data/fashion_mnist_test_vectors.csv", 10000, 784, false);
        int[] finalValLabels = dm.loadLabels("data/fashion_mnist_test_labels.csv", 10000);

        System.out.println("FINAL VALIDATION");
        neuralNetwork.validate2(finalValData, finalValLabels);

        String file_url = "test_results.txt";
        List<String> list = new ArrayList<>();
        list.add("========================");
        list.add("layers: " + Arrays.toString(hiddenLayerSizes));
        list.add("learning rate: " + learningRate);
        list.add("momentum: " + momentum);
        list.add("batch size: " + batchSize);
        list.add("epochs: " + epochs);
        list.add("dropout: " + dropout);
        list.add("decay rate: " + neuralNetwork.decayRate);
        list.add("LOSS AND ACCURACY: " + v[0] + " --- " + v[1] + "%");
        list.add("========================");
        FileUtil.writeToFile(file_url, list);
    }
}
