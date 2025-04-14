import kagglehub
import shutil
import os
from pathlib import Path

# Download latest version
def download(url: str = "atulanandjha/lfwpeople", folder_dir: Path = Path("data/raw_data")):
    path = kagglehub.dataset_download(url)

    os.makedirs(folder_dir, exist_ok = True)

    shutil.move(path, folder_dir)