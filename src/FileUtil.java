import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class FileUtil {

    // help function for initial training when tweaking parameters
    public static void writeToFile(String filePath, List<String> values) {
        // Pass 'true' to FileWriter to enable appending
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            for (String value : values) {
                writer.write(value);
                writer.newLine();
            }
            System.out.println("Values successfully appended to file: " + filePath);
        } catch (IOException e) {
            System.err.println("An error occurred while appending to the file: " + e.getMessage());
        }
    }

    //function to save to the predictions.csv file
    public static void saveLabelsToCSV(int[] predictedLabels, String fileName) {
        if (predictedLabels.length != predictedLabels.length) {
            throw new IllegalArgumentException("True labels and predicted labels arrays must have the same length.");
        }

        try (FileWriter writer = new FileWriter(fileName)) {
            for (int i = 0; i < predictedLabels.length; i++) {
                writer.append(String.valueOf(predictedLabels[i]))
                        .append("\n");
            }
            System.out.println("CSV file created successfully: " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}