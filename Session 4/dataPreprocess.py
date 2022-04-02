from collections import defaultdict
from genericpath import isfile
import os
from os import listdir
import re

def gen_data_and_vocab():
    os.makedirs('Session 4/w2v', exist_ok=True)
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'
            files = [(filename, dir_path+filename)\
                for filename in listdir(dir_path)\
                if isfile(dir_path+filename)]
            files.sort()
            label = group_id
            print(f'Processing: {group_id}-{newsgroup}')

            for filename, filepath in files:
                with open(filepath, encoding='utf8', errors='ignore') as f:
                    text = f.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(f'{label}<fff>{filename}<fff>{content}')
        return data

    word_count = defaultdict(int)
    path = './datasets/20news-bydate'
    parts = [path + '/' + dir_name + '/' for dir_name in listdir(path) \
        if not isfile(path + dir_name)]
    train_path, test_path = (parts[0], parts[1]) \
        if 'train' in parts[0] else (parts[1], parts[0])
    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()
    train_data = collect_data_from(
        parent_path = train_path,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values())\
        if freq>10]
    vocab.sort()
    with open('Session 4/w2v/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))
    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )

    with open('Session 4/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('Session 4/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))

gen_data_and_vocab()