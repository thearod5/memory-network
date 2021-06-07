"""
Responsible for generating text given some context and query.
"""
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from preprocessing.data_vectorizer import get_tokenizer, one_hot_decode
from train import CHECKPOINT_DIR

MODEL = None

STORY_MAX_LENGTH = 395
QUERY_MAX_LENGTH = 395


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model(CHECKPOINT_DIR)
    return MODEL


def create_response(context: str, query: str):
    tokenizer = get_tokenizer("sample_conversation")
    model = get_model()
    c_matrix = pad_sequences(tokenizer.texts_to_sequences([context]), maxlen=STORY_MAX_LENGTH)
    q_matrix = pad_sequences(tokenizer.texts_to_sequences([query]), maxlen=STORY_MAX_LENGTH)

    prediction = model.predict([c_matrix, q_matrix])
    return one_hot_decode(tokenizer, prediction[0])


if __name__ == "__main__":
    get_model()  # loads model
    is_exit = input("Press enter to play")

    while is_exit.lower().strip() != "exit":
        c_body = input("Context:")
        q_body = input("Query:")
        print("->", create_response(c_body, q_body))
        is_exit = input("Press enter to play")
