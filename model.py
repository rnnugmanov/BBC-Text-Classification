from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout


def create_model(max_words, embeddings_dim, max_seq_len, embeddings_matrix):
    model = Sequential()
    model.add(Embedding(max_words, embeddings_dim, input_length=max_seq_len, weights=[embeddings_matrix],
                        trainable=False))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(units=5, activation='softmax'))

    return model
