import data.DataManager;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        float[][] trainData = null;
        int[] trainLabels = null;

        try {
            int numTrainSamples = 60000;
            int numTestSamples = 10000;
            int numFeatures = 784;

            DataManager dm = new DataManager();

            trainData = DataManager.loadImageData("data/fashion_mnist_train_vectors.csv", 60000, 784);
            trainLabels = DataManager.loadLabels("data/fashion_mnist_train_labels.csv", 60000);


            DataManager.MiniBatch[] miniBatch = DataManager.getMiniBatches(trainData, trainLabels, 64);

            System.out.println(Arrays.toString(miniBatch));
            System.out.println(miniBatch.length);

            //System.out.println(testImages[0].length);
            //System.out.println(DataManager.PrintData(testImages[0]));

        } catch (IOException e) {
            e.printStackTrace();
        }

        int inputLen = 784;
        int hiddenLen = 128;
        int outputLen = 10;
        float learningRate = 0.01f;
        int batchSize = 32;
        int epochs = 10;
        int numHidden = 2;

        FeedFrwdNeuralNetwork nn = new FeedFrwdNeuralNetwork(inputLen, hiddenLen, outputLen, numHidden,learningRate);
        nn.train(trainData, trainLabels, batchSize, epochs);
    }
}