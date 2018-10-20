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


class Vocabulary:
    '''doc'''
    def __init__(self, raw_text: str) -> None:
        self.max_length = self.__max_length(raw_text)
        self.tokenizer = self.__tokenizer(raw_text)
        self.size = len(self.tokenizer.word_index.keys())

    def __max_length(self, text: str) -> int:
        text_list = re.split(r'[\t\n]+', text)
        length_list = [len(re.split(r'\s+', text)) for text in text_list]
        return max(length_list)

    def __tokenizer(
            self,
            text: str,
    ) -> tf.keras.preprocessing.text.Tokenizer:
        '''doc'''
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(
            re.split(r'[\s\t\n]', text)
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
