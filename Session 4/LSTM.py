from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from constants import MAX_DOC_LENGTH, batch_size
from tensorflow.keras.utils import to_categorical

with open('w2v/vocab-raw.txt') as f:
    vocab = dict([(word, word_ID+2) for word_ID, word in enumerate(f.read().splitlines())])
with open('w2v/20news-train-encoded.txt') as f:
    data = f.read().splitlines()
    X_train = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_train = np.array([line.split('<fff>')[0] for line in data], dtype=np.float32)
    y_train = to_categorical(y_train, 20)
with open('w2v/20news-test-encoded.txt') as f:
    data = f.read().splitlines()
    X_test = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_test = np.array([line.split('<fff>')[0] for line in data], dtype=np.float32)
    y_test = to_categorical(y_test, 20)

def model(vocab_size, embedding_size, lstm_size, output_size):
    model = keras.Sequential(
        [
            layers.Input(shape=(MAX_DOC_LENGTH, )),
            layers.Embedding(input_dim = vocab_size+2, output_dim = embedding_size, input_length=MAX_DOC_LENGTH),
            layers.LSTM(lstm_size),
            layers.Dense(output_size, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

model = model(len(vocab), 300, 50, 20)
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, epochs=5)

eval = model.evaluate(X_test, y_test)
print(f'Loss on test data: {eval[0]}, accuracy on test data: {eval[1]}')