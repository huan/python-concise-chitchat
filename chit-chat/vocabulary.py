'''
Vocabulary
'''
import re
from typing import (
    # Any,
    List,
    # Tuple,
)

import tensorflow as tf
import numpy as np

from .config import (
    GO,
    DONE,
    MAX_LENGTH,
)


class Vocabulary:
    '''doc'''
    def __init__(self, raw_text: str) -> None:
        self.tokenizer = self.__tokenizer(raw_text)
        self.size = len(self.tokenizer.word_index.keys()) + 1

    def __tokenizer(
            self,
            text: str,
    ) -> tf.keras.preprocessing.text.Tokenizer:
        '''doc'''
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(
            re.split(r'[\s\t\n]', text) + [
                GO,
                DONE,
            ]
        )
        return tokenizer

    def sentence_to_sequence(self, sentence: str) -> List[int]:
        '''doc'''
        word_list = re.split(r'\s+', sentence.strip())
        return self.tokenizer.texts_to_sequences(word_list)

    def sequence_to_sentence(self, sequence: List[int]) -> str:
        '''doc'''
        word_list = self.tokenizer.sequences_to_texts(sequence)
        # [self.index_word.get(index) for index in sequence]
        return ' '.join(word_list)

    def sentence_to_one_hot_list(self, sentence: str) -> np.ndarray:
        '''doc'''
        sequence = self.sentence_to_sequence(sentence)
        one_hot_list = np.zeros(
            (
                MAX_LENGTH,
                self.size,
            ),
            dtype=np.uint8,
        )

        for i, word_index in enumerate(sequence):
            one_hot_list[i][word_index] = 1

        return one_hot_list
