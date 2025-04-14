import tarfile
import os
from pathlib import Path 

# Path to your .tgz file
def extract_data(raw_data_dir: Path = Path("data/raw_data/3/lfw-funneled.tgz"),
                 target_dir: Path = Path('data/extracted_data')):
    tgz_path = raw_data_dir

    # Destination directory to extract to
    extract_dir = target_dir

    # Ensure the directory exists
    os.makedirs(extract_dir, exist_ok = True)

    # Extract the .tgz archive
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

    print(f"Extracted to: {extract_dir}")