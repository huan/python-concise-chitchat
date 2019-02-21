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
    EOS,
    EMBEDDING_DIM,
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

        self.encoder = ChitEncoder(embedding=self.embedding)
        # shape: [batch_size, state]

        self.repeat_vector = tf.keras.layers.RepeatVector(MAX_LEN)

        self.decoder = ChatDecoder(
            embedding=self.embedding,
            voc_size=self.voc_size,
        )
        # shape: [batch_size, max_len, voc_size]

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            training=None,
            # mask=None,
    ) -> tf.Tensor:     # shape: [batch_size, max_len, voc_size]
        '''call'''
        # import pdb; pdb.set_trace()

        outputs, hidden_state = self.encoder(
            inputs=inputs,
            training=training,
            # mask=mask,
        )

        outputs = tf.reduce_sum(outputs, 1)
        # import pdb; pdb.set_trace()
        outputs = self.repeat_vector(outputs)

        outputs, hidden_state = self.decoder(
            inputs=outputs,
            hidden_state=hidden_state,
            training=training,
            # mask=mask,
        )
        # import pdb; pdb.set_trace()
        return outputs

    def predict(
            self,
            inputs: np.ndarray,
            temperature=None,
    ) -> List[int]:
        '''doc'''
        inputs = np.expand_dims(inputs, 0)
        outputs = self(inputs)
        outputs = tf.squeeze(outputs)

        response_indices = []
        for t in range(0, MAX_LEN):
            output = outputs[t]

            indice = self.__logit_to_indice(output, temperature=temperature)

            if indice == self.word_index[EOS]:
                break

            response_indices.append(indice)

        return response_indices

    def __logit_to_indice(
            self,
            inputs,
            temperature=None,
    ) -> int:
        '''
        [vocabulary_size]
        convert one hot encoding to indice with temperature
        '''
        if temperature is None:
            indice = tf.argmax(inputs).numpy()
        else:
            prob = tf.nn.softmax(inputs / temperature).numpy()
            indice = np.random.choice(self.voc_size, p=prob)
        return indice
