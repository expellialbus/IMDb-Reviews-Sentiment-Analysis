import pathlib

from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras import Sequential
import tensorflow as tf

from preprocessing import * 

def get_model(input_dim, output_dim, dropout=0.5, layers_units=(64, 32)):
    """
    Returns the model which built according as parameters

    Parameters
    ----------
    input_dim : int
                Input dimension for embedding layer

    output_dim : int
                 Output dimension for embedding layer

    dropout : float, default=0.5
              Applied dropout level

    layers_units : tuple of integers
                   Specifies the units will be used per layer
                   Each value in the tuple corresponds to a layer with that number of units

    Returns
    -------
    model : tf.keras.models.Sequential
            A sequential model composed of specified units, layers and also a Dense layer with one unit

    """

    model = Sequential()

    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=True))
    
    for units in layers_units[:-1]:
        model.add(Bidirectional(LSTM(units, return_sequences=True, activation="tanh")))
        model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(layers_units[-1], activation="tanh")))
    model.add(Dense(1, activation="sigmoid"))

    return model

def create_full_model(model, preprocessing_layer):
    """
    Creates and returns the model which extended with preprocessing layer located at the begin

    Parameters
    ----------
    model : tf.keras.Model 
    preprocessing_layer : tf.keras.Layer

    Returns
    -------
    full_model : tf.keras.models.Sequential

    """

    full_model = Sequential()
    
    full_model.add(preprocessing_layer)
    full_model.add(model)

    return full_model

def make_prediction(model, input):
    """
    Makes a prediction and prints the positiveness percentage to the console

    Parameters
    ----------
    model : tf.keras.Model
    input : str
            Input text that will be used to make prediction

    """

    # since the model expects a batch, inputs must be turned to list
    input = tf.constant([input])

    predictions = model.predict(input)
    
    print(f"{float(predictions[0] * 100):.2f} percent positive.")

def main():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = pathlib.Path("dataset")
    
    download_dataset(url, save_path)
    text_only_set, *datasets = prepare_dataset(save_path / "aclImdb")

    vocab_size = 30000
    max_length = 400   

    (train_set, val_set, test_set), vectorizer = vectorize(datasets, text_only_set, vocab_size, max_length, num_parallel_calls=4)

    output_dim = 256
    
    model = get_model(vocab_size, output_dim)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_set.cache(), validation_data=val_set.cache(), epochs=5)

    # model results after last epoch of training
    # loss: 0.0644 - accuracy: 0.9796 - val_loss: 0.4850 - val_accuracy: 0.8562 

    # evaluation on test set 
    # loss: 0.5245 - accuracy: 0.8484
    model.evaluate(test_set)

    model = create_full_model(model, vectorizer)

    positive = "That was an awesome movie, I highly recommend you to give it a chance."
    make_prediction(model, positive) # 99.99 percent positive.

    negative = "It was a terrible movie, do not waste your time."
    make_prediction(model, negative) # 0.06 percent positive.

if __name__ == "__main__":
    main()