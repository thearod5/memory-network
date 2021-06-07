"""
The following module is responsible for turning the stories, queries, and answers into vectors.
"""
import json
import os
from typing import List

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence

from constants import PATH_TO_TOKENIZER_DATA
from models.config import MemoryNetworkConfig

TOKENIZER = None


def one_hot_encode(tokenizer: Tokenizer, texts: List[str]):
    """
    Returns a matrix containing sub-matrices for each text given where
    that sub-matrix contains a list of word one-hot encoding vectors.
    :param tokenizer: defines the word indices
    :param texts: list of texts to one hot encode
    :return:
    """
    word_sep_texts = [text_to_word_sequence(text) for text in texts]
    t_sequences = [tokenizer.texts_to_matrix(t, mode="binary") for t in word_sep_texts]
    return np.asarray(t_sequences)


def one_hot_decode(tokenizer: Tokenizer, t_matrix):
    """
    Given a matrix of one-hot encoded texts uses tokenizer to map
    one-hot encodings back to words.
    :param tokenizer: defines the word indices
    :param t_matrix: one hot encoded texts
    :return:
    """
    words = []
    for row in t_matrix:
        target_index = np.argmax(row)
        if target_index == 0:
            continue
        words.append(tokenizer.index_word[target_index])

    return " ".join(words)


def clean_doc(doc: str):
    words = []
    for w in doc.split(" "):
        if len(w) == 0 or w[0] == "<":
            continue
        words.append(w)
    return " ".join(words), len(words)


def vectorizer_data(data,
                    tokenizer: Tokenizer,
                    config: MemoryNetworkConfig,
                    update_data_lengths=True):
    """
    Transforms list of tuples of the form (story, query, answer) into vectors

    :param tokenizer:
    :param data: list of tuples containing (story, query, response)
    :param config:
    :param update_data_lengths: whether to update the input and output length of the config
    :return:
    """
    if data is None:
        return None
    stories = []
    queries = []
    responses = []

    story_len = 0
    query_len = 0
    response_len = 0
    for story, query, response in data:
        s, s_len = clean_doc(story)
        q, q_len = clean_doc(query)
        r, r_len = clean_doc(response)

        story_len = max(story_len, s_len + 2)
        query_len = max(query_len, q_len + 2)
        response_len = max(response_len, r_len + 2)

        stories.append("sos " + s + " eos")
        queries.append("sos " + q + " eos")
        responses.append("sos " + r + " eos")

    tokenizer.fit_on_texts(stories + queries + responses)
    response_encodings = one_hot_encode(tokenizer, responses)

    if update_data_lengths:
        config.story_max_length = min(config.story_max_length, story_len)
        config.query_max_length = min(config.query_max_length, query_len)
        config.answer_max_length = min(config.answer_max_length, response_len)

    story_matrix = pad_sequences(tokenizer.texts_to_sequences(stories), maxlen=config.story_max_length)
    query_matrix = pad_sequences(tokenizer.texts_to_sequences(queries), maxlen=config.query_max_length)

    if config.answer_max_length is None:
        response_matrix = np.array(response_encodings)
    else:
        response_matrix = pad_sequences(response_encodings, maxlen=config.answer_max_length)
    return story_matrix, query_matrix, response_matrix


def save_tokenizer(t, t_name: str):
    tokenizer_json = t.to_json()
    path_to_tokenizer = os.path.join(PATH_TO_TOKENIZER_DATA, t_name)
    with open(path_to_tokenizer, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def get_tokenizer(dataset_name: str):
    global TOKENIZER

    path_to_tokenizer = os.path.join(PATH_TO_TOKENIZER_DATA, dataset_name)
    if TOKENIZER is None:
        if os.path.isfile(path_to_tokenizer):
            with open(path_to_tokenizer) as f:
                tokenizer_data = json.load(f)
                TOKENIZER = tokenizer_from_json(tokenizer_data)
        else:
            TOKENIZER = Tokenizer()
            save_tokenizer(TOKENIZER, dataset_name)

    return TOKENIZER
