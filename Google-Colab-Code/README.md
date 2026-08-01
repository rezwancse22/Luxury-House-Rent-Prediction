from google.colab import files

# This will open a file picker dialog
uploaded = files.upload()

# After uploading, list the files to confirm
import os
print(os.listdir())
