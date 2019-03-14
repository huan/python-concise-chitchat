"""Pre-process"""
import gzip
import pickle
import re
import logging

from typing import (
    Dict,
    List,
    Tuple,
)
import functional
import tensorflow as tf

logging.basicConfig(format='', level=logging.INFO)

LinesTuple = Tuple[int, str]
LinesDict = Dict[int, str]

MAX_LEN = 8
# set to None to unlimit
MAX_DATASET_SIZE = None

# https://www.zhihu.com/question/20118544/answer/35639696
# 英语为母语的4岁儿童词汇量已经有5000个，8岁词汇量为10000个。
# 四级的词汇量大概为4000个左右，八级为10000个左右
VOCABULARY_SIZE = 4000


def raw_line_to_tuple(raw_line: str) -> Tuple[int, str]:
    """doc"""
    if len(raw_line.split('+++$+++')) > 4:
        line_id = int(raw_line.split()[0][1:])
        line_text = raw_line.split('+++$+++')[4:][0]
        return line_id, line_text

    return 0, ''


def preprocess_tuple(id_text_tuple: LinesTuple) -> LinesTuple:
    """doc"""
    line_id, line_text = id_text_tuple
    # new_line_text = preprocess_text(line_text)

    new_text = line_text

    new_text = re.sub('[^a-zA-Z0-9 .,?!]', ' ', new_text)
    new_text = re.sub(' +', ' ', new_text)
    new_text = re.sub(
        r'([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?',
        r'\1 \2 \3',
        new_text,
    )
    new_text = re.sub(
        r'([?!.] ).*$',
        r'\1',
        new_text,
    )
    new_text = re.sub(
        r'(\.\.\.)*$',
        r'\1',
        new_text,
    )

    if len(new_text.split()) > MAX_LEN:
        logging.info('max len exceed, skipped: {}'.format(new_text))
        return ('', '')
        # new_text = (' ').join(new_text.split()[:MAX_LEN])
        # match = re.search('[.?!]', new_text)
        # if match is not None:
        #     idx = match.start()
        #     new_text = new_text[:idx+1]

    new_text = new_text.strip().lower()

    # logging.info('before preprocess: {}'.format(line_text))
    # logging.info('after preprocess: {}'.format(new_line_text))
    return line_id, new_text


tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='',
    lower=True,
    num_words=VOCABULARY_SIZE,
)


def not_oov_tuple(id_text: LinesTuple) -> bool:
    '''doc'''
    _, text = id_text
    for word in text.split():
        if word not in tokenizer.word_index:
            return False
        if tokenizer.word_index.get(word) > tokenizer.num_words:
            return False

    # logging.info('NOT OOV for {}'.format(text))
    return True


movie_lines_raw_list: List[str] = []

# read in the Cornell Movie Dialogues data
with open(
        'data/cornell movie-dialogs corpus/movie_lines.txt',
        'r',
        encoding='iso-8859-1',
) as file:
    movie_lines_raw_list = file.read().split('\n')

if MAX_DATASET_SIZE is not None:
    movie_lines_raw_list = movie_lines_raw_list[:MAX_DATASET_SIZE]

movie_dialog_tuple_list = (
    functional.seq(movie_lines_raw_list)
    .map(raw_line_to_tuple)
    .map(preprocess_tuple)
    .filter(lambda lines_tuple: len(lines_tuple[1]) > 3)
)
logging.info('movie_dialog_tuple_list: %d', len(list(movie_dialog_tuple_list)))

#
# 2
#
# Tokenization
#

logging.info('fitting tokenizer ...')
_, all_dialog_list = zip(*movie_dialog_tuple_list)
tokenizer.fit_on_texts(all_dialog_list)
logging.info('tokenizer fitted')

with open('data/tokenizer.pkl', 'wb') as file1:
    pickle.dump(tokenizer, file1)


logging.info('filtering OOV ...')
dialog_dict: LinesDict = (
    functional.seq(movie_dialog_tuple_list)
    .filter(not_oov_tuple)
    .to_dict()
)
logging.info('OOV filtered ...')


#
# 3
#
# Build Dataset
#

question_list: List[str] = []
answer_list: List[str] = []

DATASET_FILE = 'data/dataset.txt.gz'

with gzip.open(DATASET_FILE, 'wt') as f:
    count = 0
    prev_id = 0
    for curr_id, curr_text in sorted(dialog_dict.items()):
        if prev_id + 1 == curr_id:
            prev_text = dialog_dict[prev_id]

            # question_list.append(prev_text)
            # answer_list.append(curr_text)
            f.write('{}\t{}\n'.format(
                prev_text,
                curr_text,
            ))
            count = count + 1
        prev_id = curr_id

logging.info('dataset generated: %s. total %s pairs', DATASET_FILE, count)
