from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from scipy.sparse import csr_matrix
import numpy as np

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path, encoding='unicode_escape') as f:
        d_lines = f.read().splitlines()
    with open('Session 1/TF-IDF Vectorizer/20news-bydate/words_idfs.txt', encoding='unicode_escape') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    return data, labels

def clustering_with_KMeans():
    data, labels = load_data(data_path='Session 1/TF-IDF Vectorizer/20news-bydate/full_tf-idf.txt')
    X = csr_matrix(data)
    print('='*20)
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2022
    ).fit(X)
    labels = kmeans.labels_
    print(labels)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_y = load_data('Session 1/TF-IDF Vectorizer/20news-bydate/train_tf-idf.txt')
    train_X, train_y = np.array(train_X), np.array(train_y)
    classifier = LinearSVC(
        C=10.0, # penalty coefficient
        tol=0.001, # tolerance for stopping criteria
        verbose=False # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)
    test_X, test_y = load_data('Session 1/TF-IDF Vectorizer/20news-bydate/test_tf-idf.txt')
    test_X, test_y = np.array(test_X), np.array(test_y)
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
    print(f'Accuracy: {accuracy}')

def classifying_with_kernel_SVMs():
    train_X, train_y = load_data('Session 1/TF-IDF Vectorizer/20news-bydate/train_tf-idf.txt')
    train_X, train_y = np.array(train_X), np.array(train_y)
    classifier = SVC(
        C = 50.0, # penalty coefficient
        kernel = 'rbf',
        gamma = 0.1,
        tol = 0.01, # tolerance for stopping criteria
        verbose = True # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)
    test_X, test_y = load_data('Session 1/TF-IDF Vectorizer/20news-bydate/test_tf-idf.txt')
    test_X, test_y = np.array(test_X), np.array(test_y)
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
    print(f'Accuracy: {accuracy}')

if __name__=='__main__':
    classifying_with_linear_SVMs()