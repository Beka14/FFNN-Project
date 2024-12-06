import java.io.IOException;
import java.time.LocalTime;
public class NeuralNetwork {
    private HiddenLayer[] hiddenLayers;
    private OutputLayer outputLayer;
    private float learningRate;

    private static float[][] validateData;
    private static int[] validateLabels;
    private float decayRate = 0.0f;

    public NeuralNetwork(int inputLen, int[] hiddenLayerSizes, int outputLen, float learningRate, float momentum) {
        // init all hidden layers and output layers and set learning rate

        hiddenLayers = new HiddenLayer[hiddenLayerSizes.length];

        hiddenLayers[0] = new HiddenLayer(inputLen, hiddenLayerSizes[0], learningRate, momentum);
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            hiddenLayers[i] = new HiddenLayer(hiddenLayerSizes[i - 1], hiddenLayerSizes[i], learningRate, momentum);
        }

        outputLayer = new OutputLayer(hiddenLayerSizes[hiddenLayerSizes.length - 1], outputLen, learningRate, momentum);

        this.learningRate = learningRate;
    }

    //forwardpass logic
    public float[][] forwardPass(float[][] inputBatch) {
        float[][] hiddenOutput = inputBatch;
        for (HiddenLayer layer : hiddenLayers) {
            hiddenOutput = layer.forwardPass(hiddenOutput);
        }
        return outputLayer.forwardPass(hiddenOutput);
    }

    // training function
    public void trainBatch(float[][] trainData, int[] trainLabels, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // compute the decayed learning rate
            float decayingLR = (float) (learningRate * Math.exp(-decayRate * epoch));
            System.out.println("LEARNING RATE: " + decayingLR);

            // create mini-batches
            DataManager.MiniBatch[] miniBatches = DataManager.getMiniBatches(trainData, trainLabels, batchSize);

            float totalLoss = 0;

            for (DataManager.MiniBatch batch : miniBatches) {
                float[][] batchData = batch.data;
                float[][] batchLabels = DataManager.toOneHot(batch.labels, outputLayer.outputLen);

                // update learning rate for all layers
                for (HiddenLayer layer : hiddenLayers) {
                    layer.setLearningRate(decayingLR);
                }
                outputLayer.setLearningRate(decayingLR);

                // forward pass through the entire batch
                float[][] hiddenOutput = batchData;
                for (HiddenLayer layer : hiddenLayers) {
                    hiddenOutput = layer.forwardPass(hiddenOutput);
                }
                float[][] predictions = outputLayer.forwardPass(hiddenOutput);

                // compute batch loss
                totalLoss += outputLayer.crossEntropyLossBatch(predictions, batchLabels);

                // backward pass for the entire batch
                float[][] outputGradients = outputLayer.backProp(batchLabels);
                float[][] gradients = outputGradients;
                for (int i = hiddenLayers.length - 1; i >= 0; i--) {
                    gradients = hiddenLayers[i].backProp(gradients);
                }

                // update parameters for each layer
                for (HiddenLayer layer : hiddenLayers) {
                    layer.updateParameters();
                }
                outputLayer.updateParametersMomentum(batchSize);
            }

            System.out.printf("Epoch %d - Loss: %.4f%n", epoch + 1, totalLoss / trainData[0].length);
        }
    }


    // validation function, the data go through forward pass and loss is computed
    public float[] validate(float[][] valData, int[] valLabels, String file_name) {
        float[][] predictions = forwardPass(valData);
        int correct = 0;
        float valLoss = 0;
        int[] predicted = new int[valLabels.length];

        float[][] targets = DataManager.toOneHot(valLabels, outputLayer.outputLen);

        for (int i = 0; i < valData.length; i++) {
            int predictedClass = getArgMax(predictions[i]);
            predicted[i] = predictedClass;
            if (predictedClass == valLabels[i]) {
                correct++;
            }

            valLoss += outputLayer.crossEntropyLossBatch(predictions, targets);
        }

        FileUtil.saveLabelsToCSV(predicted, file_name); // create the predictions file(s)

        float accuracy = (float) correct / valData.length;
        System.out.printf("Validation Accuracy: %.2f%%%n", accuracy * 100);
        return new float[] {valLoss, accuracy * 100};
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
        float learningRate = 0.008f;
        float momentum = 0.9f;
        int batchSize = 32;
        int epochs = 10;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLen, hiddenLayerSizes, outputLen, learningRate, momentum);
        neuralNetwork.decayRate = 0.02F;

        float[][] trainData;
        int[] trainLabels;

        DataManager dm = new DataManager();

        float[][] images = dm.loadImageData("data/fashion_mnist_train_vectors.csv", 60000, 784, true);
        int[] labels = dm.loadLabels("data/fashion_mnist_train_labels.csv", 60000);

        DataManager.DataSplit split = DataManager.splitData(images, labels, 0.8f);
        trainData = split.trainData;
        trainLabels = split.trainLabels;;

        validateData = split.valData;
        validateLabels = split.valLabels;

        System.out.println("start:" + LocalTime.now());
        neuralNetwork.trainBatch(trainData, trainLabels, batchSize, epochs);

        float[] v = neuralNetwork.validate(validateData, validateLabels, "train_predictions.csv");

        float[][] finalValData = dm.loadImageData("data/fashion_mnist_test_vectors.csv", 10000, 784, false);
        int[] finalValLabels = dm.loadLabels("data/fashion_mnist_test_labels.csv", 10000);

        System.out.println("FINAL VALIDATION");
        neuralNetwork.validate(finalValData, finalValLabels, "test_predictions.csv");
        System.out.println("end:" + LocalTime.now());
    }
}
