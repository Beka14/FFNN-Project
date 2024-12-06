#!/bin/bash

module add java

SRC_DIR="src"
BIN_DIR="bin"
MAIN_CLASS="NeuralNetwork"
DATA_DIR="data"

mkdir -p "$BIN_DIR"

echo "Compiling all Java files..."
javac -d "$BIN_DIR" "$SRC_DIR"/*.java
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi
echo "Compilation successful."

echo "Executing the program..."
nice -n 19 java -cp "$BIN_DIR" "$MAIN_CLASS"
if [ $? -ne 0 ]; then
    echo "Execution failed. Exiting."
    exit 1
fi

echo "Execution completed successfully and files with predictions created."
