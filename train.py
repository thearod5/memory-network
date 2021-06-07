from loaders.data_loader import DataLoader
from models.builder import create_memory_network
from models.config import MemoryNetworkConfig

N_EPOCHS = 30
BATCH_SIZE = 32

if __name__ == "__main__":
    config = MemoryNetworkConfig()
    config.vocab_size = 1500
    config.embedding_size = 50
    config.dropout_rate = 0.3
    config.n_lstm_nodes = 64
    config.story_max_length = 500
    config.query_max_length = 500
    config.answer_max_length = 500  # seq2seq models
    config.n_decoder_lstm = 3
    config.n_encoder_lstm = 3

    d_name = "sample_conversation"
    loader = DataLoader(d_name)
    train_data, test_data = loader.get_vectorized_data(config)
    train_stories, train_queries, train_answers = train_data
    test_stories, test_queries, test_answers = test_data

    memory_network = create_memory_network(config)
    memory_network.load_weights(f"weights/{d_name}.h5")

    history = memory_network.fit([train_stories, train_queries],
                                 train_answers,
                                 BATCH_SIZE,
                                 N_EPOCHS,
                                 validation_data=([test_stories, test_queries], test_answers))

    memory_network.save(f"weights/{d_name}.h5")
    print("saved!")
