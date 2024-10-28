package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

public class DataManager {
    private static final int rows = 28;
    private static final int cols = 28;

    public static float[][] data;
    public static int[] labels;

    public final String trainVectorsPath = "data/fashion_mnist_train_vectors.csv";
    public final String testVectorsPath = "data/fashion_mnist_test_vectors.csv";
    public final String trainLabelsPath = "data/fashion_mnist_train_labels.csv";
    public final String testLabelsPath = "data/fashion_mnist_test_labels.csv";

    public static float[][] loadImageData(String filePath, int numRows, int numCols) throws IOException {
        data = new float[numRows][numCols];

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;

            while ((line = br.readLine()) != null && row < numRows) {
                String[] values = line.split(",");
                for (int col = 0; col < values.length; col++) {
                    data[row][col] = Float.parseFloat(values[col]) / 255.0f; // Normalize pixel values
                }
                row++;
            }
        }

        return data;
    }

    public static int[] loadLabels(String filePath, int numLabels) throws IOException {
        labels = new int[numLabels];

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int i = 0;

            while ((line = br.readLine()) != null && i < numLabels) {
                labels[i] = Integer.parseInt(line.trim());
                i++;
            }
        }

        return labels;
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

    public static String PrintData(float[] data) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                sb.append(data[i * 28 + j]).append(",");
            }
            sb.append("\n");
        }
        return sb.toString();
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////// MINI BATCH CLASS /////////////////////////////////////////////////

    public static class MiniBatch {
        public float[][] data;
        public float[] labels;

        public MiniBatch(float[][] data, float[] labels) {
            this.data = data;
            this.labels = labels;
        }
    }

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
            float[] Blabels = new float[currBSize];

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
