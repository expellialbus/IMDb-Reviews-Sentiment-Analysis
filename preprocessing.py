import requests
import tarfile
import pathlib
import shutil
import random
import os

from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(compressed, save_path.parent)

def prepare_dataset(dataset_path, batch_size=32, val_ratio=0.1, return_text_only_train_set=True):
    """
    Reads dataset from the specified path and then creates the validation set
    according as speficied size

    Parameters
    ----------
    dataset_path : str, pathlib.Path
                   Dataset path which contains the downloaded dataset

    batch_size : int, default=32
                 Batch size for creating datasets

    val_ratio : float, default=0.1 (10% of train set)
                Validation set ratio which is used to derive validation set 
                from both Train set and Test set

    return_text_only_train_set : bool, optional, default=True
                                 Condition to return a text only dataset (without labels) version of train set
                                 Text only dataset may be useful to adapt/fit vectorizer or tokenizer
                                 

    Returns
    -------
    train_set : BatchDataset, (dtype=tf.string, dtype=tf.int32)
    val_set :  BatchDataset, (dtype=tf.string, dtype=tf.int32)
    test_set : BatchDataset, (dtype=tf.string, dtype=tf.int32)
    text_only_set : MapDataset, dtype=tf.string

    Note
    ----
    text_only_set : only returned if return_text_only_train_set equals to True
    train_set : contains 22500 data sample belonging to 2 classes
    val_set : contains 4913 data sample belonging to 2 classes
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

    if return_text_only_train_set:
        text_only_set = train_set.map(lambda x, y: x)

        return text_only_set, train_set, val_set, test_set
    else:
        return train_set, val_set, test_set

def vectorize(datasets, 
              text_only_dataset, 
              vocab_size,
              max_length, 
              output_mode="int", 
              num_parallel_calls=None, 
              return_vectorizer=True):
    """
    Vectorizes the datasets according as the specified parameters

    Parameters
    ----------
    datasets : iterable object
               Contains datasets as order unaware

    text_only_dataset : tf.data.Dataset, tf.string
                        A dataset just contains inputs (not labels)
    
    vocab_size : int, default=3e4
                 Vocabulary size
                
    max_length : int, default=400
                 Max length per review (just works if output mode is "int")
                 A review longer than max length will be truncated
                 Lower than max length will be padded with zeros

    output_mode : str, default=int
                  Generated dataset type

    num_parallel_calls : int, optional
                         Number of core will be used during text vectorization process

    return_vectorizer : bool, default=True
                        Condition to return vectorizer object
                        If it is True, vectorizer object used during vectorization process will be returned
                        Otherwise the object is not returned 

    Returns 
    -------
    vectorized_datasets : list object (at the order of passed datasets)
    text_vectorization : tf.keras.layers.TextVectorization (according as specified arguments), optional

    Notes
    -----
    For more info about default arguments, see the tf.keras.layers.TextVectorization layer 

    """

    text_vectorization = TextVectorization(max_tokens=vocab_size, 
                                           output_sequence_length=max_length, 
                                           output_mode=output_mode)

    text_vectorization.adapt(text_only_dataset)
    
    vectorized_datasets = list()
    for dataset in datasets:
        vectorized_datasets.append(dataset.map(lambda x, y: (text_vectorization(x), y), 
                                               num_parallel_calls=num_parallel_calls).prefetch(1))
        
    if return_vectorizer:
        return vectorized_datasets, text_vectorization
    else:
        return vectorized_datasets