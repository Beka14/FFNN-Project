import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DataManager {

    public static float[][] data;
    public static int[] labels;

    public static float mean;
    public static float std;

    public static float[][] loadImageData(String filePath, int numRows, int numCols, boolean training) throws IOException {
        float[][] data = new float[numRows][numCols];

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                for (int col = 0; col < values.length; col++) {
                    data[row][col] = Float.parseFloat(values[col]) / 255.0f; //normalizing the data first
                }
                row++;
            }
        }

        // calc the global mean and standard deviation only if training and then use it later for testing
        if (training) {
            mean = calculateMean(data);
            std = calculateStd(data, mean);
        }

        // normalize the data using the global mean and standard deviation
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                data[i][j] = (data[i][j] - mean) / std;
            }
        }

        return data;
    }

    public static int[] loadLabels(String filePath, int numLabels) throws IOException {
        labels = new int[numLabels];

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int i = 0;

            while ((line = br.readLine()) != null) {
                labels[i] = Integer.parseInt(line.trim());
                i++;
            }
        }

        return labels;
    }

    public static float[][] toOneHot(int[] labels, int numClasses) {
        float[][] oneHot = new float[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            oneHot[i][labels[i]] = 1.0f;
        }
        return oneHot;
    }

    private static float calculateMean(float[][] data) {
        float sum = 0.0f;
        int count = 0;

        for (float[] row : data) {
            for (float value : row) {
                sum += value;
                count++;
            }
        }

        return sum / count;
    }

    private static float calculateStd(float[][] data, float mean) {
        float sumSquaredDifferences = 0.0f;
        int count = 0;

        for (float[] row : data) {
            for (float value : row) {
                sumSquaredDifferences += Math.pow(value - mean, 2);
                count++;
            }
        }

        return (float) Math.sqrt(sumSquaredDifferences / count);
    }


    // shuffle indices instead of data
    public static int[] generateShuffledIndices(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }

        Random random = new Random();
        for (int i = size - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            int temp = indices[index];
            indices[index] = indices[i];
            indices[i] = temp;
        }

        return indices;
    }

    // split data and labels for training and testing
    public static DataSplit splitData(float[][] data, int[] labels, float splitRatio) {
        int totalSize = data.length;
        int trainSize = (int) (totalSize * splitRatio);

        float[][] trainData = new float[trainSize][data[0].length];
        int[] trainLabels = new int[trainSize];

        float[][] valData = new float[totalSize - trainSize][data[0].length];
        int[] valLabels = new int[totalSize - trainSize];

        int[] shuffledIndices = generateShuffledIndices(totalSize);

        for (int i = 0; i < trainSize; i++) {
            trainData[i] = data[shuffledIndices[i]];
            trainLabels[i] = labels[shuffledIndices[i]];
        }

        for (int i = trainSize; i < totalSize; i++) {
            valData[i - trainSize] = data[shuffledIndices[i]];
            valLabels[i - trainSize] = labels[shuffledIndices[i]];
        }

        return new DataSplit(trainData, trainLabels, valData, valLabels);
    }

    public static int[] shuffle(int[] array){
        Random r = new Random();
        for(int i=array.length-1; i>0;i--){
            int index = r.nextInt(i+1);
            int a = array[index];
            array[index] = array[i];
            array[i] = a;
        }

        return array;
    }

    public static class DataSplit {
        public float[][] trainData;
        public int[] trainLabels;
        public float[][] valData;
        public int[] valLabels;

        public DataSplit(float[][] trainData, int[] trainLabels, float[][] valData, int[] valLabels) {
            this.trainData = trainData;
            this.trainLabels = trainLabels;
            this.valData = valData;
            this.valLabels = valLabels;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////// MINI BATCH CLASS /////////////////////////////////////////////////

    public static class MiniBatch {
        public float[][] data;
        public int[] labels;

        public MiniBatch(float[][] data, int[] labels) {
            this.data = data;
            this.labels = labels;
        }
    }

    //creates minibatches
    public static MiniBatch[] getMiniBatches(float[][] data, int[] labels, int batchSize) {
        int numExamples = data.length;
        int numFeatures = data[0].length;

        int numBatches = (int) Math.ceil((double) numExamples / batchSize);

        MiniBatch[] batches = new MiniBatch[numBatches];

        int[] indices = new int[numExamples];
        for (int i = 0; i < numExamples; i++) {
            indices[i] = i;
        }

        indices = shuffle(indices);

        for (int batch = 0; batch < numBatches; batch++) {
            int start = batch * batchSize;
            int end = Math.min(start + batchSize, numExamples);
            int currBSize = end - start;

            float[][] Bdata = new float[currBSize][numFeatures];
            int[] Blabels = new int[currBSize];

            for (int i = 0; i < currBSize; i++) {
                int index = indices[start + i];
                Bdata[i] = data[index];
                Blabels[i] = labels[index];
            }

            batches[batch] = new MiniBatch(Bdata, Blabels);
        }

        return batches;
    }



}
