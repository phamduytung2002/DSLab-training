import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from constants import MAX_DOC_LENGTH
from tensorflow.keras.utils import to_categorical

with open('w2v/vocab-raw.txt', encoding='unicode_escape') as f:
    vocab = dict([(word, word_ID+2) for word_ID, word in enumerate(f.read().splitlines())])
with open('w2v/20news-train-encoded.txt', encoding= 'unicode_escape') as f:
    data = f.read().splitlines()
    X_train = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_train = np.array([line.split('<fff>')[0] for line in data], dtype=np.float32)
    y_train = tf.one_hot(indices=y_train, depth=20, dtype=tf.float32)
with open('w2v/20news-test-encoded.txt', encoding= 'unicode_escape') as f:
    data = f.read().splitlines()
    X_test = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_test = np.array([line.split('<fff>')[0] for line in data], dtype=np.float32)
    y_test = tf.one_hot(indices=y_test, depth=20, dtype=tf.float32)

def model(vocab_size, embedding_size, lstm_size, output_size):
    model = keras.Sequential(
        [
            layers.Input(shape=(MAX_DOC_LENGTH, )),
            layers.Embedding(input_dim = vocab_size+2, output_dim = embedding_size, input_length=MAX_DOC_LENGTH, mask_zero=True),
            layers.LSTM(lstm_size),
            layers.Dense(output_size, activation="softmax"),
        ]
    )
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])
    return model

model = model(len(vocab), 300, 50, 20)
model.summary()

epochs = 10

for epochID in range(epochs):
    print(f'epoch {epochID}')
    model.fit(X_train, y_train, batch_size=50, epochs=1)
    eval = model.evaluate(X_test, y_test)
    print(f'Loss on test data: {eval[0]}, accuracy on test data: {eval[1]}')