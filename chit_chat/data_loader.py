'''
data loader
'''
import gzip
import re
from typing import (
    # Any,
    List,
    Tuple,
)

import tensorflow as tf
import numpy as np

from .config import (
    EOS,
    MAX_LEN,
)

DATASET_URL = 'https://github.com/huan/concise-chit-chat/releases/download/v0.0.1/dataset.txt.gz'
DATASET_FILE_NAME = 'concise-chit-chat-dataset.txt.gz'


class DataLoader():
    '''data loader'''

    def __init__(self) -> None:
        print('DataLoader', 'downloading dataset from:', DATASET_URL)
        dataset_file = tf.keras.utils.get_file(
            DATASET_FILE_NAME,
            origin=DATASET_URL,
        )
        print('DataLoader', 'loading dataset from:', dataset_file)

        # dataset_file = './data/dataset.txt.gz'

        # with open(path, encoding='iso-8859-1') as f:
        with gzip.open(dataset_file, 'rt') as f:
            self.raw_text = f.read().lower()

        self.queries, self.responses \
            = self.__parse_raw_text(self.raw_text)
        self.size = len(self.queries)

    def get_batch(
            self,
            batch_size=32,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        '''get batch'''
        # print('corpus_list', self.corpus)
        batch_indices = np.random.choice(
            len(self.queries),
            size=batch_size,
        )
        batch_queries = self.queries[batch_indices]
        batch_responses = self.responses[batch_indices]

        return batch_queries, batch_responses

    def __parse_raw_text(
            self,
            raw_text: str
    ) -> Tuple[List[List[str]], List[List[str]]]:
        '''doc'''
        query_list = []
        response_list = []

        for line in raw_text.strip('\n').split('\n'):
            query, response = line.split('\t')
            query, response = self.preprocess(query), self.preprocess(response)
            query_list.append('{} {}'.format(query, EOS))
            response_list.append('{} {}'.format(response, EOS))

        return np.array(query_list), np.array(response_list)

    def preprocess(self, text: str) -> str:
        '''doc'''
        new_text = text

        new_text = re.sub('[^a-zA-Z0-9 .,?!]', ' ', new_text)
        new_text = re.sub(' +', ' ', new_text)
        new_text = re.sub(
            r'([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?',
            r'\1 \2 \3',
            new_text,
        )
        if len(new_text.split()) > MAX_LEN:
            new_text = (' ').join(new_text.split()[:MAX_LEN])
            match = re.search('[.?!]', new_text)
            if match is not None:
                idx = match.start()
                new_text = new_text[:idx+1]

        new_text = new_text.strip().lower()

        return new_text
