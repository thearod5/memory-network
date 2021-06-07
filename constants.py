import os

PATH_TO_CHAT_BOT = os.path.join(os.path.dirname(__file__))
PATH_TO_TOKENIZER_DATA = os.path.join(PATH_TO_CHAT_BOT, "t_data")

# sample data
PATH_TO_CHAT_BOT_DATA = os.path.join(PATH_TO_CHAT_BOT, "data")
PATH_TO_SAMPLE_DATA = os.path.join(PATH_TO_CHAT_BOT_DATA, 'en-10k', 'qa1_single-supporting-fact_{}.txt')

# model weights
PATH_TO_WEIGHTS = os.path.join(PATH_TO_CHAT_BOT, "weights")

# conversation data
PATH_TO_CONVERSATION_DATA = os.path.join(PATH_TO_CHAT_BOT_DATA, "conversation")
