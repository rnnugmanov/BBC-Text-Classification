import sys
import logging as log

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from bbc_text_classification.parse_news import parse_bbc_news
from train import MAX_WORDS, MAX_SEQ_LEN
from utils import parse_to_labels, tokenize

logger = log.getLogger()
log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO, stream=sys.stdout)


SAVED_DIR = r"C:\Users\Ruslan\PycharmProjects\pythonProject\bbc_text_classification\saved_model_dir"
DATA_DIR = r"C:\Users\Ruslan\Desktop\datasets\bbc_text_classification\bbc"
BASE_URL = "https://www.bbc.com"
PAGE_PATH = "https://www.bbc.com/news/stories"


if '__main__' == __name__:
    logger.info(f"Loading TF model from {SAVED_DIR}")
    predictor = tf.keras.models.load_model(SAVED_DIR)

    logger.info(f"Loading text dataset to create tokenizer")
    dataset, labels = parse_to_labels(DATA_DIR)
    _, tokenizer = tokenize(dataset, MAX_WORDS)

    logger.info(f"Parsing news from {PAGE_PATH}")
    texts = parse_bbc_news(BASE_URL, PAGE_PATH)

    classes = ['business', 'entertainment', 'politics', 'sport', 'tech']
    for link, text in texts.items():
        logger.info(f"Making classification for {link}")
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, MAX_SEQ_LEN)
        prediction = predictor.predict(text)
        predicted_class = np.argmax(prediction)

        logger.info(f" Predicted class: {classes[predicted_class]},"
                    f" with probability {int(prediction[0][predicted_class] * 100)}%")
