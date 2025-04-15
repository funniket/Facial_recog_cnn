import os
import tarfile
import kagglehub

# Download the dataset using kagglehub.
download_path = kagglehub.dataset_download("atulanandjha/lfwpeople")
print("Path to downloaded dataset:", download_path)

# Check if the download_path is a directory. If so, search for a .tgz or .tar.gz file.
if os.path.isdir(download_path):
    candidate = None
    for file in os.listdir(download_path):
        if file.endswith('.tgz') or file.endswith('.tar.gz'):
            candidate = os.path.join(download_path, file)
            break
    if candidate is None:
        raise Exception("No .tgz or .tar.gz file found in the downloaded directory.")
    tgz_path = candidate
else:
    tgz_path = download_path

print("Using tgz file:", tgz_path)

# Define the directory where you want to extract the dataset.
extract_dir = "data"
os.makedirs(extract_dir, exist_ok=True)

# Open and extract the tgz file.
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print("Dataset extracted to directory:", extract_dir)
