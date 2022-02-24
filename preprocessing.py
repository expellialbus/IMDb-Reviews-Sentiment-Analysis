import requests
import tarfile
import pathlib
import shutil
import random
import os

from tensorflow import keras

def download_dataset(url, save_path=""):
    """
    Downloads a given file and saves it to specified path

    Parameters
    ----------
    url : str
          Download source

    save_path : str, pathlib.Path, optional
                File path to save the downloaded file
                If it is not specified, downloaded file will be saved current working directory
                Directory does not have to exist

    """

    file_name = url.rsplit('/', maxsplit=1)[1] # gets file name from url
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_path /= file_name
    
    response = requests.get(url)

    # saves downloaded file
    with open(save_path, "wb") as file:
        file.write(response.content)

    # extracts the downloaded tar file
    with tarfile.open(save_path) as compressed:
        compressed.extractall(save_path.parent)

def prepare_dataset(dataset_path, batch_size=32, val_ratio=0.1):
    """
    Reads dataset from the specified path and then creates the validation set
    according as speficied size

    Parameters
    ----------
    dataset_path : str, pathlib.Path
                   Dataset path which contains the downloaded dataset

    batch_size : int, optional
                 Batch size for creating datasets, default=32

    val_ratio : float, optional
               Validation set ratio which is used to derive validation set 
               from both Train set and Test set

    Returns
    -------
    train_set : BatchDataset, dtype=tf.int32
    val_set :  BatchDataset, dtype=tf.int32
    test_set : BatchDataset, dtype=tf.int32

    Note
    ----
    train_set : contains 22500 data sample belonging to 2 classes
    val_set : contains 4910 data sample belonging to 2 classes
    test_set : contains 22500 data sample belonging to 2 classes

    """

    root = pathlib.Path(dataset_path)

    # deletes all files in depth 0 (inside root directory) and 1 (inside sub directory)
    for base in root.iterdir():
        if base.is_file(): 
            base.unlink()
            continue

        for sub in base.iterdir():
            if sub.is_file():
                sub.unlink()

    shutil.rmtree(root / "train" / "unsup") # this folder also is unnecessary (contains unlabelled data)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    for category in ("neg", "pos"):
        val_cat = val_dir / category
        os.makedirs(val_cat)
        
        for folder in (train_dir, test_dir):
            folder_cat = folder / category
            files = list(folder_cat.rglob("*"))

            random.Random(42).shuffle(files)
            num_val_samples = int(val_ratio * len(files))

            for fname in files[:num_val_samples]:
                shutil.move(fname, val_cat / fname.name)

    train_set = keras.utils.text_dataset_from_directory(train_dir, batch_size=batch_size)
    val_set = keras.utils.text_dataset_from_directory(val_dir, batch_size=batch_size)
    test_set = keras.utils.text_dataset_from_directory(test_dir, batch_size=batch_size)

    return train_set, val_set, test_set