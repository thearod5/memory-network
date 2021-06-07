"""
Responsible for defining a memory network model configurable to multi-answer responses.
"""
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, LSTM, Permute, RepeatVector, \
    TimeDistributed, add, \
    concatenate, dot
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from tensorflow.python.keras.layers import Attention

from models.config import MemoryNetworkConfig


def create_embedding(input_dim,
                     output_dim,
                     dropout_rate,
                     momentum,
                     input_length=None):
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=input_dim,
                                  output_dim=output_dim,
                                  input_length=input_length))
    input_encoder_m.add(BatchNormalization(momentum=momentum))
    input_encoder_m.add(Dropout(dropout_rate))
    return input_encoder_m


def create_memory_network_encoder(input_sequence: Input,
                                  question: Input,
                                  config: MemoryNetworkConfig):
    input_encoder_m = create_embedding(input_dim=config.vocab_size,
                                       output_dim=config.embedding_size,
                                       dropout_rate=config.dropout_rate,
                                       momentum=config.momentum)
    input_encoder_c = create_embedding(input_dim=config.vocab_size,
                                       output_dim=config.query_max_length,
                                       dropout_rate=config.dropout_rate,
                                       momentum=config.momentum)
    question_encoder = create_embedding(input_dim=config.vocab_size,
                                        output_dim=config.embedding_size,
                                        dropout_rate=config.dropout_rate,
                                        momentum=config.momentum,
                                        input_length=config.query_max_length)

    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, story_maxlen, query_maxlen)
    match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
    match = Activation('softmax')(match)

    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

    # concatenate the response vector with the question vector sequence
    encoder = concatenate([response, question_encoded])
    for i in range(config.n_encoder_lstm - 1):
        encoder = LSTM(config.n_lstm_nodes, return_sequences=True)(encoder)
    encoder = LSTM(config.n_lstm_nodes)(encoder)
    encoder = Dropout(config.dropout_rate)(encoder)
    return encoder


def create_sequential_decoder(decoder_input, config: MemoryNetworkConfig):
    decoder = RepeatVector(config.answer_max_length)(decoder_input)
    for i in range(config.n_decoder_lstm):
        decoder = LSTM(config.n_lstm_nodes, return_sequences=True)(decoder)
    decoder = BatchNormalization(momentum=config.momentum)(decoder)
    decoder = Dropout(config.dropout_rate)(decoder)
    decoder = TimeDistributed(Dense(config.vocab_size, activation='softmax'))(decoder)
    return decoder


def create_sequential_decoder_with_attention(decoder_input, config: MemoryNetworkConfig):
    decoder = RepeatVector(config.answer_max_length)(decoder_input)
    decoder = LSTM(config.n_lstm_nodes, return_sequences=True)(decoder)
    decoder = Attention()(decoder)
    decoder = BatchNormalization(momentum=config.momentum)(decoder)
    decoder = TimeDistributed(Dense(config.vocab_size, activation='softmax'))(decoder)
    return decoder


def create_memory_network(config: MemoryNetworkConfig):
    context_input = Input((config.story_max_length,))
    query_input = Input((config.query_max_length,))

    answer = create_memory_network_encoder(context_input,
                                           query_input,
                                           config)
    decoder = create_sequential_decoder(answer, config)

    model = Model([context_input, query_input], decoder)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
