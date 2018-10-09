"""Pre-process"""
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
    """doc"""
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

    new_text = '{} {} {}'.format(
        BOS,
        new_text.strip(),
        EOS,
    ).lower()

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


movie_lines_raw_list: List[str] = []

# read in the Cornell Movie Dialogues data
with open(
        'data/cornell movie-dialogs corpus/movie_lines.txt',
        'r',
        encoding='iso-8859-1',
) as file:
    movie_lines_raw_list = file.read().split('\n')

# FIXME: for quick debug
movie_lines_raw_list = movie_lines_raw_list[:10000]

movie_dialog_tuple_list = (
    functional.seq(movie_lines_raw_list)
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
_, all_dialog_list = zip(*movie_dialog_tuple_list)
tokenizer.fit_on_texts(all_dialog_list)
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


# https://stackoverflow.com/a/9001529/1123955

print('sorting ...')
sorted_dialog_dict = collections.OrderedDict(
    sorted(
        dialog_dict.items()
    )
)
print('sorted ...')


question_list: List[str] = []
answer_list: List[str] = []

prev_id = 0
for curr_id, curr_text in sorted_dialog_dict.items():
    if prev_id + 1 == curr_id:
        prev_text = sorted_dialog_dict[prev_id]

        question_list.append(prev_text)
        answer_list.append(curr_text)

    prev_id = curr_id

question_sequence_list = tokenizer.texts_to_sequences(question_list)
answer_sequence_list = tokenizer.texts_to_sequences(answer_list)

np.save('data/questions', question_sequence_list)
np.save('data/answers', answer_sequence_list)


print('questions/answers: #{}/{}'.format(
    len(question_sequence_list),
    len(answer_sequence_list),
))

for i in range(3):
    print('question {}: {}'.format(i, question_list[i]))
    print('question sequence {}: {}'.format(i, question_sequence_list[i]))
    print('answer {}: {}'.format(i, answer_list[i]))
    print('answer sequence {}: {}'.format(i, answer_sequence_list[i]))

