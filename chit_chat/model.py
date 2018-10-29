'''doc'''
from typing import (
    List,
    # Tuple,
)

import tensorflow as tf
import numpy as np

from .chit_encoder import ChitEncoder
from .chat_decoder import ChatDecoder
from .config import (
    DONE,
    GO,
    MAX_LEN,
)
from .vocabulary import Vocabulary


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary: Vocabulary,
    ) -> None:
        super().__init__()

        self.word_index = vocabulary.tokenizer.word_index
        self.index_word = vocabulary.tokenizer.index_word
        self.voc_size = vocabulary.size

        # [batch_size, max_len] -> [batch_size, max_len, embedding_dim]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.voc_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
        )

        self.encoder = ChitEncoder()
        # shape: [batch_size, state]

        self.decoder = ChatDecoder(self.voc_size)
        # shape: [batch_size, max_len, voc_size]

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            teacher_forcing_targets: List[List[int]]=None,  # shape: [batch_size, max_len]
            training=None,
            mask=None,
    ) -> tf.Tensor:     # shape: [batch_size, max_len, voc_size]
        '''call'''
        state = self.encoder(inputs)
        return self.decoder(
            state,
            training=training,
            teacher_forcing_targets=teacher_forcing_targets,
        )


    def predict(self, inputs: np.ndarray, temperature=1.) -> List[int]:
        '''doc'''

        inputs = np.expand_dims(inputs, 0)
        outputs = self(inputs)
        outputs = tf.squeeze(outputs)

        response_indices = []
        for t in range(1, MAX_LEN):
            output = outputs[t]

            indice = self.__logit_to_indice(output, temperature=temperature)

            if indice == self.word_index[DONE]:
                break

            response_indices.append(indice)

        return response_indices

    def __logit_to_indice(
            self,
            inputs,
            temperature=1.,
    ) -> int:
        '''
        [vocabulary_size]
        convert one hot encoding to indice with temperature
        '''
        inputs = tf.squeeze(inputs)
        prob = tf.nn.softmax(inputs / temperature).numpy()
        indice = np.random.choice(self.voc_size, p=prob)
        return indice
