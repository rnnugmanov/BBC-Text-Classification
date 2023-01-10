from bbc_text_classification.utils.model import create_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# main paths
DATA_DIR = r"C:\Users\Ruslan\Desktop\datasets\bbc_text_classification\bbc"
EMBEDDINGS_PATH = r"C:\Users\Ruslan\Desktop\datasets\glove_embeddings\glove.6B.300d.txt"
BEST_RESULTS_DIR = r"C:\Users\Ruslan\PycharmProjects\pythonProject\bbc_text_classification\best_model"
LOG_DIR = r"C:\Users\Ruslan\PycharmProjects\pythonProject\bbc_text_classification\log_dir"

# dataset params
MAX_WORDS = 10000
MAX_SEQ_LEN = 600
TRAIN_SAMPLES_NUM = 2000
VAL_SAMPLES_NUM = 225

# train params
EPOCHS = 100
BATCH_SIZE = 32

if __name__ == '__main__':
    texts, labels = parse_to_labels(DATA_DIR)
    embeddings_index = load_embeddings(EMBEDDINGS_PATH)
    sequences, tokenizer = tokenize(texts, MAX_WORDS)
    embedding_matrix = create_embeddings_matrix(embeddings_index, tokenizer.word_index, MAX_WORDS)
    sequences = pad_sequences(sequences, MAX_SEQ_LEN)

    train_x, train_y, val_x, val_y = shuffle_and_divide_data(sequences, labels, TRAIN_SAMPLES_NUM, VAL_SAMPLES_NUM)
    model = create_model(max_words=MAX_WORDS, max_seq_len=MAX_SEQ_LEN,
                         embeddings_dim=embedding_matrix.shape[1], embeddings_matrix=embedding_matrix)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    check_point = tf.keras.callbacks.ModelCheckpoint(BEST_RESULTS_DIR, monitor='loss', save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    model.fit(train_x, train_y,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(val_x, val_y),
              callbacks=[check_point, tb_callback])

