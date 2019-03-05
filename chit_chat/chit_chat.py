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
    PAD,
    SOS,
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

        self.indice_sos = self.word_index[SOS]
        self.indice_eos = self.word_index[EOS]
        self.indice_pad = self.word_index[PAD]

        # [batch_size, max_len] -> [batch_size, max_len, embedding_dim]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.voc_size,
            output_dim=EMBEDDING_DIM,
        )

        self.encoder = ChitEncoder()
        # shape: [batch_size, state]

        self.decoder = ChatDecoder(
            voc_size=self.voc_size,
        )
        # shape: [batch_size, max_len, voc_size]

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            training=False,
            teacher_forcing_targets=None,
            # mask=None,
    ) -> tf.Tensor:     # shape: [batch_size, max_len, voc_size]
        '''call'''
        # import pdb; pdb.set_trace()

        inputs = tf.convert_to_tensor(inputs)
        batch_size = tf.shape(inputs)[0]

        inputs = self.embedding(inputs)

        encoder_outputs, encoder_hidden_state = self.encoder(
            inputs=inputs,
            # training=training,
            # mask=mask,
        )

        # import pdb; pdb.set_trace()

        batch_sos_one_hot = tf.ones([batch_size, 1, 1]) \
            * [tf.one_hot(self.indice_sos, self.voc_size)]

        decoder_output = batch_sos_one_hot
        decoder_state = encoder_hidden_state

        outputs = tf.zeros([batch_size, 0, self.voc_size])

        # print('*'*20)
        for t in range(0, MAX_LEN):
            # import pdb; pdb.set_trace()
            if training and teacher_forcing_targets is not None:
                target_indice = tf.expand_dims(
                    teacher_forcing_targets[:, t], axis=-1)
            else:
                target_indice = tf.argmax(decoder_output, axis=-1)

            # print('{} {}'.format(t, ' '.join([self.index_word[i] for i in target_indice.numpy()[0]])))
            decoder_inputs = self.embedding(target_indice)

            decoder_output, decoder_state = self.decoder(
                inputs=decoder_inputs,
                initial_state=decoder_state,
            )

            outputs = tf.concat([outputs, decoder_output], axis=1)

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

            if indice == self.indice_eos:
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
