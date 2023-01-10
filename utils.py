from tqdm import tqdm
import os
import numpy as np
import logging as log
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def parse_to_labels(data_path):
    labels = []
    texts = []
    dir_to_iterate = [path for path in os.listdir(data_path) if path[-4:].lower() != '.txt']

    for label, directory in enumerate(dir_to_iterate):
        path_to_iterate = os.path.join(data_path, directory)

        log.info(f"Loading files from {path_to_iterate} directory")
        for file_name in tqdm(os.listdir(path_to_iterate)):
            if file_name[-4:] == '.txt':
                text_file = os.path.join(path_to_iterate, file_name)
                text = open(text_file)
                texts.append(text.read())
                text.close()

                labels.append(label)

    return texts, to_categorical(np.asarray(labels))


def load_embeddings(emb_path):
    embeddings_index = {}
    f = open(emb_path, encoding='utf-8')

    for line in tqdm(f):
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype=np.float32)
        embeddings_index[word] = embedding

    f.close()

    return embeddings_index


def tokenize(texts: list, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    return sequences, tokenizer


def create_embeddings_matrix(embeddings: dict, word_to_index_map: dict, max_words):
    embeddings_dim = list(embeddings.values())[0].shape[0]
    all_embeddings = np.stack(embeddings.values())
    embeddings_mean = all_embeddings.mean()
    embeddings_std = all_embeddings.std()

    nb_words = min(max_words, len(word_to_index_map))

    embeddings_matrix = np.random.normal(embeddings_mean, embeddings_std, (nb_words, embeddings_dim))

    for word, i in word_to_index_map.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix


def shuffle_and_divide_data(sequences, labels, train_samples_num, validation_samples_num):
    indices = np.arange(sequences.shape[0])
    np.random.shuffle(indices)
    sequences = sequences[indices]
    labels = labels[indices]

    train_x = sequences[:train_samples_num]
    train_y = labels[:train_samples_num]
    val_x = sequences[train_samples_num:train_samples_num + validation_samples_num]
    val_y = labels[train_samples_num:train_samples_num + validation_samples_num]

    return train_x, train_y, val_x, val_y
