# IMDb Reviews Sentiment Analysis

This project aims to classify Imdb reviews according to their sentiments. The way that it tries to solve this problem is to train a Bidirectional LSTM with the dataset in [this link](https://ai.stanford.edu/~amaas/data/sentiment). The dataset can also be found under the _dataset_ folder of the project.

Files and folders of the project.

> [model](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/tree/master/model)

This folder contains a zip file for model (this model file is fully saved model architecture,not just weights).

> [dataset](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/tree/master/dataset)

As mentioned above, this folder contains the dataset download from the provided link.

> [preprocessing.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/preprocessing.py)

Contains codes for preprocessing the dataset.

> [train.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/train.py)

Contains codes for building and training the the model.

# How Things Work

## What does the [preprocessing.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/preprocessing.py) file do?

In short, as the name suggests, the [preprocessing.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/preprocessing.py) file contains some preprocess functions for the [dataset](https://github.com/recep-yildirimIMDb-Reviews-Sentiment-Analysis/blob/master/dataset/aclImdb_v1.tar.gz).

It offers a method to download the [dataset](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/dataset/aclImdb_v1.tar.gz). After complete dowloading, it decompresses the file and save it to the path specified by a parameter. Another function inside of the [preprocessing.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/preprocessing.py) file prepares (removes unnecessary files/folder, splits the train data to a validation data by the ratio that controlled with a parameter) the dataset that read from the specified path and returns it as splitted to train, test and validation sets. Final function, basically applies the vectorization operation to the dataset.

**Note:** \*For more information about the functions in the [preprocessing.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/preprocessing.py) file, please read the docs in the file.

## How about the [train.py](https://github.com/recep-yildirim/IMDb-Reviews-Sentiment-Analysis/blob/master/train.py) file?

It contains a function that creates and returns the model, personalized by the parameters. Also another utility function that can be useful when you need to serve your model in a deployment environment. This function basically merges the preprocessing layers and the model that you created into a new model that can handle the preprocessing operations on the fly (without the need for any preprocessing operations) and returns this brand new model. It is very useful when you want to maintain your preprocessing layers in one place.

# Test Results

## Metrics

\
**Train**

_Loss_: 0.0644

_Accuracy_: 0.9796

\
**Validation**

_Loss_: 0.4850

_Accuracy_: 0.8562

\
**Test**

_Loss_: 0.5245

_Accuracy_: 0.8484

## Example Inputs

_input_: "That was an awesome movie, I highly recommend you to give it a chance."

_result_: 99.99 percent positive.

\
_input_: "It was a terrible movie, do not waste your time."

_result_: 0.06 percent positive.
