import os

from constants import PATH_TO_WEIGHTS


def get_model_versions():
    previous_runs = os.listdir(PATH_TO_WEIGHTS)
    version_numbers = [int(r.split("_")[-1].split(".")[0]) for r in previous_runs if
                       r[0] != "." and MODEL_ID in r]
    return version_numbers


def load_recent_weights(model):
    version_numbers = get_model_versions()
    version_number = max(version_numbers)
    path_to_file = os.path.join(PATH_TO_WEIGHTS, f"{MODEL_ID}_{version_number}.h5")
    model.load_weights(path_to_file)
    return model


def save_model(config, model):
    version_numbers = get_model_versions()
    new_version = 0 if len(version_numbers) == 0 else max(version_numbers) + 1
    path_to_file = os.path.join(PATH_TO_WEIGHTS, f"{MODEL_ID}_{new_version}.h5")
    model.save(path_to_file)


class MemoryNetworkConfig:
    def __init__(self):  # TODO: set defaults
        self.vocab_size = None
        self.embedding_size = None
        self.dropout_rate = None
        self.n_lstm_nodes = None
        self.story_max_length = None
        self.query_max_length = None
        self.answer_max_length = None  # seq2seq models
        self.dataset_name = None
        self.n_decoder_lstm = 3
        self.n_encoder_lstm = 3
        self.model_id = "memory_networks"
        self.momentum = 0.7
