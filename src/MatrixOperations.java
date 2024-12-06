public class MatrixOperations {

    // class for all matrix operations that i use

    public static float[][] matrixMultiply(float[][] A, float[][] B) {
        int rowsA = A.length, colsA = A[0].length, colsB = B[0].length;
        float[][] result = new float[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    public static float[][] addMatrix(float[][] A, float[][] B) {
        int rows = A.length, cols = A[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    }

    public static float[] addVector(float[] A, float[] B) {
        int length = A.length;
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = A[i] + B[i];
        }
        return result;
    }

    public static float[][] transpose(float[][] A) {
        int rows = A.length, cols = A[0].length;
        float[][] result = new float[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }


    public static float[][] elementWiseMultiply(float[][] A, float[][] B) {
        int rows = A.length, cols = A[0].length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] * B[i][j];
            }
        }
        return result;
    }


    public static float[] sumAlongAxis(float[][] A, int axis) {
        if (axis == 0) { // column-wise sum
            int cols = A[0].length;
            float[] result = new float[cols];
            for (float[] row : A) {
                for (int j = 0; j < cols; j++) {
                    result[j] += row[j];
                }
            }
            return result;
        } else if (axis == 1) { // row-wise sum
            int rows = A.length;
            float[] result = new float[rows];
            for (int i = 0; i < rows; i++) {
                for (float val : A[i]) {
                    result[i] += val;
                }
            }
            return result;
        }
        return null;
    }


    public static float[][] subtractMatrix(float[][] A, float[][] B) {
        int rows = A.length, cols = A[0].length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        return result;
    }


    public static float[] subtractVector(float[] A, float[] B) {
        int length = A.length;
        float[] result = new float[length];

        for (int i = 0; i < length; i++) {
            result[i] = A[i] - B[i];
        }
        return result;
    }


    public static float[][] scalarMultiply(float[][] A, float scalar) {
        int rows = A.length, cols = A[0].length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] * scalar;
            }
        }
        return result;
    }


    public static float[] scalarMultiply(float[] A, float scalar) {
        int length = A.length;
        float[] result = new float[length];

        for (int i = 0; i < length; i++) {
            result[i] = A[i] * scalar;
        }
        return result;
    }
}

