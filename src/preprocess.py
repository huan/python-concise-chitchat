"""get data"""
import pickle
import re
from typing import (
    Dict,
    List,
    Tuple,
)
import collections
import numpy as np
import functional
import tensorflow as tf


LinesTuple = Tuple[int, str]
LinesDict = Dict[int, str]

BOS = 'BOS'
EOS = 'EOS'
UNK = 'UNK'

MAX_LEN = 12

# https://www.zhihu.com/question/20118544/answer/35639696
# 英语为母语的4岁儿童词汇量已经有5000个，8岁词汇量为10000个。
# 四级的词汇量大概为4000个左右，八级为10000个左右
VOCABULARY_SIZE = 1000

#
# 1
#
# Load Curpos Lines to Dict
# Preprocessing
#


def preprocess_text(text: str) -> str:
    """
    do some basic preprocessing, filter out dialogues with more than 12 words,
    and in the 12 or lesser words, take only the characters
    till one of '!' or '.' or '?' comes
    """

    new_text = text

    new_text = re.sub('[^a-zA-Z0-9 .,?!]', ' ', new_text)
    new_text = re.sub(' +', ' ', new_text)
    new_text = re.sub(
        '([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?',
        r'\1 \2 \3',
        new_text,
    )
    if len(new_text.split()) > MAX_LEN:
        new_text = (' ').join(new_text.split()[:MAX_LEN])
        match = re.search('[.?!]', new_text)
        if match is not None:
            idx = match.start()
            new_text = new_text[:idx+1]

    # Add BOS(Beginning of Sentence) and EOS(End of Sentence)
    new_text = new_text.strip().lower()
    new_text = '{} {} {}'.format(BOS.lower(), new_text, EOS.lower())

    return new_text


def raw_line_to_tuple(raw_line: str) -> Tuple[int, str]:
    """doc"""
    if len(raw_line.split('+++$+++')) > 4:
        line_id = int(raw_line.split()[0][1:])
        line_text = raw_line.split('+++$+++')[4:][0]
        return line_id, line_text

    return 0, ''


def preprocess_tuple(id_text: LinesTuple) -> LinesTuple:
    """doc"""
    line_id, line_text = id_text
    new_line_text = preprocess_text(line_text)
    # print('before preprocess: {}'.format(line_text))
    # print('after preprocess: {}'.format(new_line_text))
    return line_id, new_line_text


tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='',
    lower=True,
    num_words=VOCABULARY_SIZE,
    oov_token=UNK.lower(),
)


def not_oov_tuple(id_text: LinesTuple) -> bool:
    '''doc'''
    _, text = id_text
    for word in text.split():
        if word not in tokenizer.word_index:
            return False
        if tokenizer.word_index.get(word) > tokenizer.num_words:
            return False

    # print('NOT OOV for {}'.format(text))
    return True


movie_raw_lines_list: List[str] = []

# read in the Cornell Movie Dialogues data
with open(
        'data/cornell movie-dialogs corpus/movie_lines.txt',
        'r',
        encoding='iso-8859-1',
) as file:
    movie_raw_lines_list = file.read().split('\n')

# FIXME: for quick debug
movie_raw_lines_list = movie_raw_lines_list[:50000]

movie_dialog_tuple_list = (
    functional.seq(movie_raw_lines_list)
    .map(raw_line_to_tuple)
    .map(preprocess_tuple)
    .filter(lambda lines_tuple: len(lines_tuple[1]) > 0)
)
print('movie_dialog_tuple_list: {}'.format(len(list(movie_dialog_tuple_list))))

#
# 2
#
# Tokenization
#

print('fitting tokenizer ...')
_, movie_dialog_list = zip(*movie_dialog_tuple_list)
tokenizer.fit_on_texts(movie_dialog_list)
print('tokenizer fitted')

with open('data/tokenizer.pkl', 'wb') as file1:
    pickle.dump(tokenizer, file1)


print('filtering OOV ...')
dialog_dict: LinesDict = (
    functional.seq(movie_dialog_tuple_list)
    .filter(not_oov_tuple)
    .to_dict()
)
print('OOV filtered ...')


#
# 3
#
# Build Dataset
#


# sort the dialogues into the proper sequence
# based on the line number 'L...' in the data
# https://stackoverflow.com/a/9001529/1123955

print('sorting ...')
sorted_dialog_dict = collections.OrderedDict(
    sorted(
        dialog_dict.items()
    )
)
print('sorted ...')


current_sentence_list: List[str] = []
next_sentence_list: List[str] = []

prev_id = 0
for curr_id, curr_text in sorted_dialog_dict.items():
    if prev_id + 1 == curr_id:
        prev_text = sorted_dialog_dict[prev_id]

        current_sentence_list.append(prev_text)
        next_sentence_list.append(curr_text)

    prev_id = curr_id

current_sentence_sequence_list = tokenizer.texts_to_sequences(current_sentence_list)
next_sentence_sequence_list = tokenizer.texts_to_sequences(next_sentence_list)

np.save('data/current_sentence', current_sentence_sequence_list)
np.save('data/next_sentence', next_sentence_sequence_list)


print('current/next: #{}/{}'.format(
    len(current_sentence_sequence_list),
    len(next_sentence_sequence_list),
))

for i in range(3):
    print('current dialog {}: {}'.format(i, current_sentence_list[i]))
    print('current sequence {}: {}'.format(i, current_sentence_sequence_list[i]))
    print('next dialog {}: {}'.format(i, next_sentence_list[i]))
    print('next sequence {}: {}'.format(i, next_sentence_sequence_list[i]))

