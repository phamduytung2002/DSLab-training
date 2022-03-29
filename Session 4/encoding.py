from constants import MAX_DOC_LENGTH, unknown_ID, padding_ID

def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID+2) for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])\
            for line in f.read().splitlines()]
            
    encoded_data = []
    current_label = -1
    for document in documents:
        if document[0] != current_label:
            current_label = document[0]
            print(current_label)
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(f'{vocab[word]}')
            else:
                encoded_text.append(f'{unknown_ID}')
            
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(f'{padding_ID}')
        encoded_data.append(f'{label}<fff>{doc_id}<fff>{sentence_length}<fff>'\
             + ' '.join(encoded_text))
    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + 'encoded.txt'
    with open(f'{dir_name}'+file_name, 'w') as f:
        f.write('\n'.join(encoded_data))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(f'{dir_name}/{file_name}', 'w') as f:
        f.write('\n'.join(encoded_data))

encode_data('w2v/20news-train-raw.txt', 'w2v/vocab-raw.txt')
encode_data('w2v/20news-test-raw.txt', 'w2v/vocab-raw.txt')