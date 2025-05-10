#!/bin/bash
echo "Creating model directories..."
mkdir -p model/variables

# Download gdown for Google Drive downloads
pip install gdown

# Download the model variables file
echo "Downloading model variables file..."
FILE_ID="1OJS9QKFsvmKx43Ry55R01lu0bjw_WV-d"
OUTPUT_PATH="model/variables/variables.data-00000-of-00001"
gdown --id $FILE_ID --output $OUTPUT_PATH

echo "Model download complete!"

# Verify the file was downloaded
if [ -f "$OUTPUT_PATH" ]; then
    echo "Model file successfully downloaded to $OUTPUT_PATH"
    ls -la $OUTPUT_PATH
else
    echo "ERROR: Failed to download model file"
    exit 1
fi
