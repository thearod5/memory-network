"""
Responsible for generating text given some context and query.
"""
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from preprocessing.data_vectorizer import get_tokenizer, one_hot_decode
from train import get_model_path

MODEL = None

STORY_MAX_LENGTH = 395
QUERY_MAX_LENGTH = 395


def get_model(m_name: str):
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model(get_model_path(m_name))
    return MODEL


def create_response(model, context: str, query: str):
    tokenizer = get_tokenizer("sample_conversation")
    c_matrix = pad_sequences(tokenizer.texts_to_sequences([context]), maxlen=STORY_MAX_LENGTH)
    q_matrix = pad_sequences(tokenizer.texts_to_sequences([query]), maxlen=STORY_MAX_LENGTH)

    prediction = model.predict([c_matrix, q_matrix])
    return one_hot_decode(tokenizer, prediction[0])


if __name__ == "__main__":
    m_name = "mn"
    model = get_model(m_name)  # loads model
    is_exit = input("Press enter to play")

    while is_exit.lower().strip() != "exit":
        c_body = input("Context:")
        q_body = input("Query:")
        print("->", create_response(model, c_body, q_body))
        is_exit = input("Press enter to play")
