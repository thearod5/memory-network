from loaders.data_loader import DataLoader
from models.builder import create_memory_network
from models.config import MemoryNetworkConfig

N_EPOCHS = 1
BATCH_SIZE = 32
if __name__ == "__main__":
    config = MemoryNetworkConfig()
    config.vocab_size = 1500
    config.embedding_size = 50
    config.dropout_rate = 0.30
    config.n_lstm_nodes = 64
    config.story_max_length = 500
    config.query_max_length = 500
    config.answer_max_length = 500  # seq2seq models
    config.n_decoder_lstm = 4

    train_data, test_data = DataLoader("sample_conversation").get_vectorized_data(config)
    train_stories, train_queries, train_answers = train_data
    memory_network = create_memory_network(config)
    memory_network.fit([train_stories, train_queries], train_answers, validation_data=([test_data[:2], test_data[2]]))
