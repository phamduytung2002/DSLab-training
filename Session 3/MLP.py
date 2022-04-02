import tensorflow.compat.v1 as tf
import numpy as np
import random

NUM_CLASSES = 20

class MLP:
  def __init__(self, vocab_size, hidden_size):
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size

  def build_graph(self):
    self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
    self._real_Y = tf.placeholder(tf.int32, shape=[None,])

    weights_1 = tf.get_variable(
        name = 'weights_input_hidden',
        shape = (self._vocab_size, self._hidden_size),
        initializer = tf.random_normal_initializer(seed = 2022)
    )
    biases_1 = tf.get_variable(
        name = 'biases_input_hidden',
        shape = (self._hidden_size),
        initializer = tf.random_normal_initializer(seed = 2022)
    )
    weights_2 = tf.get_variable(
        name = 'weights_hidden_output',
        shape = (self._hidden_size, NUM_CLASSES),
        initializer = tf.random_normal_initializer(seed = 2022)
    )
    biases_2 = tf.get_variable(
        name = 'biases_hidden_output',
        shape = (NUM_CLASSES),
        initializer = tf.random_normal_initializer(seed = 2022)
    )
    hidden = tf.matmul(self._X, weights_1) + biases_1
    hidden = tf.sigmoid(hidden)
    logits = tf.matmul(hidden, weights_2) + biases_2

    labels_one_hot = tf.one_hot(indices=self._real_Y, depth=NUM_CLASSES,
                                dtype=tf.float32)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                   logits = logits)
    loss = tf.reduce_mean(loss)
    probs = tf.nn.softmax(logits)
    predicted_labels = tf.argmax(probs, axis=1)
    predicted_labels = tf.squeeze(predicted_labels)
    return predicted_labels, loss

  def trainer(self, loss, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op

class DataReader:
    def __init__ (self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2022)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        return self._data[start:end], self._labels[start:end]

def load_dataset():
    train_data_reader = DataReader(
        data_path = 'Session 1/TF-IDF Vectorizer/data/train-tf-idf.txt',
        batch_size = 50,
        vocab_size = vocab_size
    )
    test_data_reader = DataReader(
        data_path = 'Session 1/TF-IDF Vectorizer/data/test-tf-idf.txt',
        batch_size = 50,
        vocab_size = vocab_size
    )
    return train_data_reader, test_data_reader

def save_parameters(name, value, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1:
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[row]]) for row in range(value.shape[0])])

    with open('Session 3/saved-paras/' + filename, 'w') as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open("Session 3/saved-paras/" + filename) as f:
        lines = f.read().splitlines()
    if len(lines)==1:
        value = [float(number) for number in lines[0].split(',')]
    else:
        value = [[float(number) for number in lines[row].split(',')] for row in range(len(lines))]
    return value

if __name__ == '__main__':
    tf.disable_v2_behavior()
    with open('Session 1/TF-IDF Vectorizer/data/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.1)

    with tf.Session() as sess:
        train_data_reader, test_data_reader = load_dataset()
        step, MAX_STEP = 0, 1000

        sess.run(tf.global_variables_initializer())
        with open('Session 3/log.txt', 'a') as f:
            while step < MAX_STEP:
                train_data, train_labels = train_data_reader.next_batch()
                plabels_eval, loss_eval, _ = sess.run(
                    [predicted_labels, loss, train_op],
                    feed_dict = {
                        mlp._X: train_data,
                        mlp._real_Y: train_labels
                    }
                )
                step += 1
                f.write(f'step:{step}, loss:{loss_eval}\n')
        
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            save_parameters(
                name = variable.name,
                value = variable.eval(),
                epoch = train_data_reader._num_epoch
            )
        
    test_data_reader = DataReader(
        data_path = 'Session 1/TF-IDF Vectorizer/data/test-tf-idf.txt',
        batch_size = 50,
        vocab_size = vocab_size
    )
    with tf.Session() as sess:
        epoch = train_data_reader._num_epoch
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(
                predicted_labels,
                feed_dict = {
                    mlp._X: test_data,
                    mlp._real_Y: test_labels
                }
            )
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))
            if test_data_reader._batch_id == 0:
                break
        with open('Session 3/log.txt', 'a') as f:
            f.write(f'Epoch: {epoch}\n')
            f.write(f'Accuracy on test data: {num_true_preds/len(test_data_reader._data)}\n')