package data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class FileUtil {

    /**
     * Appends a list of values to a text file.
     * Each value is written on a new line.
     *
     * @param filePath the path to the file
     * @param values   the list of values to append
     */
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
}