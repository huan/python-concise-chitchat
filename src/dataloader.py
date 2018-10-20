'''
data loader
'''
from typing import (
    # Any,
    List,
    Tuple,
)

import numpy as np

DATASET_URL = 'https://github.com/zixia/concise-chit-chat/releases/download/v0.0.1/dataset.txt.gz'
DATASET_FILE_NAME = 'concise-chit-chat-dataset.txt'


class DataLoader():
    '''data loader'''

    def __init__(self) -> None:
        # path = tf.keras.utils.get_file(DATASET_FILE_NAME, origin=DATASET_URL)

        # XXX
        path = './data/dataset.txt'
        # print('path', path)

        with open(path, encoding='iso-8859-1') as f:
            self.raw_text = f.read().lower()

        # line_list = self.raw_text_to_line_list(self.raw_text)

        # XXX
        # print('raw_text', raw_text.split('\n'))

        self.left_sentence_list, self.right_sentence_list \
            = self.raw_text_to_sentence_list_tuple(self.raw_text)

        # left_sequence_list, right_sequence_list \
        #     = self.raw_text_to_sequence_list_tuple(raw_text)

        # self.corpus_left = self.sequence_list_to_numpy(left_sequence_list)
        # self.corpus_right = self.sequence_list_to_numpy(right_sequence_list)

        self.size = len(self.left_sentence_list)

    def get_batch(self, batch_size=32) -> Tuple[List[str], List[str]]:
        '''get batch'''
        # print('corpus_list', self.corpus)
        random_index_list = np.random.choice(
            self.size,
            size=batch_size,
        )
        left = self.left_sentence_list[random_index_list]
        right = self.right_sentence_list[random_index_list]

        # [batch_size, seq_length], [batch_size, ]
        # print(left.shape)
        # print(right.shape)

        return left.tolist(), right.tolist()

    def raw_text_to_sentence_list_tuple(
            self,
            raw_text: str
    ) -> Tuple[List[str], List[str]]:
        '''doc'''
        line_list = raw_text.strip('\n').split('\n')
        left_right_list = [
            (
                line.split('\t')[0],
                line.split('\t')[1],
            )
            for line in line_list
        ]
        left_sentence_list, right_sentence_list = zip(*left_right_list)
        return (
            np.array(left_sentence_list),
            np.array(right_sentence_list),
        )
