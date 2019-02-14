'''doc'''
import re
from typing import (
    List,
)

import tensorflow as tf

from .config import (
    EOS,
    MAX_LEN,
    PAD,
)


class Vocabulary:
    '''voc'''
    def __init__(self, text: str) -> None:
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts(
            [EOS, PAD] + re.split(
                r'[\n\s\t]',
                text,
            )
        )
        # additional 1 for the index 0
        self.size = 1 + len(self.tokenizer.word_index.keys())

    def texts_to_padded_sequences(
            self,
            text_list: List[List[str]],
    ) -> tf.Tensor:
        '''doc'''
        sequence_list = self.tokenizer.texts_to_sequences(text_list)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequence_list,
            maxlen=MAX_LEN,
            truncating='post',
            value=self.tokenizer.word_index.get(PAD),
        )

        return padded_sequences
