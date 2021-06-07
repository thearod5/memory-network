"""
Responsible for testing that vectorizing words is accurately being computed forwards
and backwards
"""

import unittest

from keras_preprocessing.text import Tokenizer

from preprocessing.data_vectorizer import one_hot_decode, one_hot_encode


class TestVectorization(unittest.TestCase):
    def test_sample(self):
        tokenizer = Tokenizer()
        texts = ['The cat sat on the mat.', 'The dog ate my homework.']

        tokenizer.fit_on_texts(texts)
        t_matrix = one_hot_encode(tokenizer, texts)

        sentence = one_hot_decode(tokenizer, t_matrix[0])
        self.assertEqual(sentence, "the cat sat on the mat")

        sentence = one_hot_decode(tokenizer, t_matrix[1])
        self.assertEqual(sentence, "the dog ate my homework")
