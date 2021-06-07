from tensorflow.python.keras.preprocessing.text import Tokenizer

from loaders.conversation import get_conversation_data, get_sample_conversation_data
from loaders.sample import load_sample_data
from models.config import MemoryNetworkConfig
from preprocessing.data_vectorizer import one_hot_decode, save_tokenizer, vectorizer_data

data_loader_mapping = {
    "sample_conversation": get_sample_conversation_data,
    "conversation": get_conversation_data,
    "sample": load_sample_data
}


class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.tokenizer: Tokenizer = None

    def get_vectorized_data(self, config: MemoryNetworkConfig):
        train_data, test_data = data_loader_mapping[self.dataset_name]()
        self.tokenizer = Tokenizer(num_words=config.vocab_size)
        v_train = vectorizer_data(train_data,
                                  self.tokenizer,
                                  config, update_data_lengths=True)
        v_test = vectorizer_data(test_data,
                                 self.tokenizer,
                                 config, update_data_lengths=False)
        save_tokenizer(self.tokenizer, self.dataset_name)
        return v_train, v_test

    def decode_one_hot(self, sequence):
        return one_hot_decode(self.tokenizer, sequence)

    def decode_sequence(self, sequence):
        words = []
        for word_index in sequence:
            if word_index == 0:
                continue
            words.append(self.tokenizer.index_word[word_index])
        return " ".join(words)
