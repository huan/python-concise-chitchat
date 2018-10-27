# 序列到序列（SEQ2SEQ）

Sequence to Sequence学习最早由

这篇文章主要是提供了一种崭新的RNN Encoder-Decoder算法，并且将其应用于机器翻译中。

这种

所谓的RNN Encoder-Decoder结构，简单的来说就是算法包含两部分，一个负责对输入的信息进行Encoding，将输入转换为向量形式。

然后由Decoder对这个向量进行解码，还原为输出序列。

而RNN Encoder-Decoder结构就是编码器与解码器都是使用RNN算法，一般为LSTM。

LSTM的优势在于处理序列，它可以将上文包含的信息保存在隐藏状态中，这样就提高了算法对于上下文的理解能力。

Encoder与Decoder各自可以算是单独的模型，一般是一层或多层的LSTM。



序列到序列模型（Sequence to Sequence, SEQ2SEQ）是一种基于RNN的Encoder-Decoder结构，它也是现在谷歌应用于线上机器翻译的算法，翻译质量已经和人类水平不相上下。

这里的Encoder-Decoder结构，简单的来说就是算法包含两部分，一个负责对输入的信息进行Encoding，将输入转换为向量形式；然后由Decoder对这个向量进行解码，还原为输出序列。

关于SEQ2SEQ的原理，可以参考：

- To be added
- To be added

这里，我们使用SEQ2SEQ来实现一个闲聊（Chit Chat）对话机器人。除了闲聊机器人（输入一句话，输出一句回复）之外，它也可以被直接应用于解决其他类似问题，比如：翻译（输入一句英文，输出一句中文）、摘要（输入一篇文章，输出一份总结）、作诗（输入几个关键字，输出一首短诗）、对对联（输入上联，输出下联），等等。

这个任务对比与之前的RNN尼采风格文本生成，区别在于我们预测的不再是文本的连续字母概率分布，而是通过一个序列，来预测另外一个对应的序列。举例来说，针对一句常见的打招呼：

```text
How are you
```

这个句子（序列）一共有3个单词。当我们听到这个由3个单词组成的句子后，根据我们的习惯，我们最倾向与回复的一句话是"Fine thank you"。我们希望建立这样一个模型：

1. Encoder: 输入num_batch个由编码后单词组成的，长度为max_length的序列（单词个数不足max_length的句子，用代表PADDING的编码补齐，一般为0），输入张量形状为[num_batch, max_length]，输出这些序列的上下文张量，形状为([num_encoder_units], [num_encoder_units])；
2. Decoder: 输入上下文张量，形状为([num_encoder_units], [num_encoder_units])，同时输入时刻为t的单词编码（纯量）；输出新的上下文张量，形状为([num_encoder_units], [num_encoder_units])，以及预测的时刻为t+1的单词编码（纯量）；
3. 执行时，我们将一句话输入Encoder，得到上下文张量；然后将上下文张量，和代表句子开始t0时刻的字符编码（我们下面用`GO`表示句子开始，用`DONE`表示句子结束），输入Decoder，即可获得新的上下文张量，并预测得到句子的第一个单词编码。然后，我们将新的上下文张量，和预测得到的单词编码再重复送入Decoder，即可解码第二个单词，然后滚雪球式地生成第三个单词，第四个单词等等，直到单词数量达到max_length，或得到代表句子结束的单词`DONE`，即可完成单词序列的生成任务。

首先，还是实现一个简单的 ``DataLoader`` 类来读取文本，

```py
DATASET_URL = 'https://github.com/zixia/concise-chit-chat/releases/download/v0.0.1/dataset.txt.gz'
DATASET_FILE_NAME = 'concise-chit-chat-dataset.txt.gz'

class DataLoader():
    def __init__(self) -> None:
        dataset_file = tf.keras.utils.get_file(DATASET_FILE_NAME, origin=DATASET_URL)
        with gzip.open(dataset_file, 'rt') as f:
            self.raw_text = f.read().lower()
        self.queries, self.responses = self.__parse_raw_text(self.raw_text)
        self.size = len(self.queries)

    def get_batch(self, batch_size=32):
        batch_indices = np.random.choice(self.size, size=batch_size)
        return self.queries[batch_indices], self.responses[batch_indices]

    def preprocess(self, text):
        text = re.sub('[^a-zA-Z0-9 .,?!]', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?', r'\1 \2 \3', text)
        if len(text.split()) > MAX_LENGTH:
            text = (' ').join(text.split()[:MAX_LENGTH])
            match = re.search('[.?!]', text)
            if match is not None:
                idx = match.start()
                text = text[:idx+1]
        return text.strip().lower()

    def __parse_raw_text(self, raw_text):
        query_list, response_list = [], []
        for line in raw_text.strip('\n').split('\n'):
            query, response = line.split('\t')
            query, response = self.preprocess(query), self.preprocess(response)
            query_list.append('{} {} {}'.format(GO, query, DONE))
            response_list.append('{} {} {}'.format(GO, response, DONE))
        return np.array(query_list), np.array(response_list)
```

其次，我们还需要基于`DataLoader`加载的文本数据，建立一个词汇表`Vocabulary`来负责管理以下5项任务：

1. 将所有单词和标点符号进行编码；
2. 记录词汇表大小；
3. 建立单词到编码数字，以及编码数字到单词的映射字典；
4. 负责将文本句子转化为填充后的编码序列，形状为[batch_size, max_length]；

```python
class Vocabulary:
    def __init__(self, text):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts([GO, DONE] + re.split(r'[\s\t\n]', text))
        self.size = 1 + len(self.tokenizer.word_index.keys())

    def texts_to_padded_sequences(self, text_list):
        sequence_list = self.tokenizer.texts_to_sequences(text_list)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequence_list, maxlen=MAX_LENGTH, padding='post', truncating='post')
        return padded_sequences
```

接下来进行模型的实现。在 ``__init__`` 方法中我们实例化一个常用的 ``BasicLSTMCell`` 单元，以及一个线性变换用的全连接层，我们首先对序列进行One Hot操作，即将编码i变换为一个n维向量，其第i位为1，其余均为0。这里n为字符种类数num_char。变换后的序列张量形状为[num_batch, seq_length, num_chars]。接下来，我们将序列从头到尾依序送入RNN单元，即将当前时间t的RNN单元状态 ``state`` 和t时刻的序列 ``inputs[:, t, :]`` 送入RNN单元，得到当前时间的输出 ``output`` 和下一个时间t+1的RNN单元状态。取RNN单元最后一次的输出，通过全连接层变换到num_chars维，即作为模型的输出。

.. figure:: ../_static/image/model/rnn_single.jpg
    :width: 30%
    :align: center

    ``output, state = self.cell(inputs[:, t, :], state)`` 图示

.. figure:: ../_static/image/model/rnn.jpg
    :width: 50%
    :align: center

    RNN流程图示

具体实现如下：

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 7-21

训练过程与前节基本一致，在此复述：

- 从DataLoader中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 59-69

关于文本生成的过程有一点需要特别注意。之前，我们一直使用 ``tf.argmax()`` 函数，将对应概率最大的值作为预测值。然而对于文本生成而言，这样的预测方式过于绝对，会使得生成的文本失去丰富性。于是，我们使用 ``np.random.choice()`` 函数按照生成的概率分布取样。这样，即使是对应概率较小的字符，也有机会被取样到。同时，我们加入一个 ``temperature`` 参数控制分布的形状，参数值越大则分布越平缓（最大值和最小值的差值越小），生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 23-28

通过这种方式进行“滚雪球”式的连续预测，即可得到生成文本。

.. literalinclude:: ../_static/code/zh/model/rnn/rnn.py
    :lines: 71-78

生成的文本如下::

    diversity 0.200000:
    conserted and conseive to the conterned to it is a self--and seast and the selfes as a seast the expecience and and and the self--and the sered is a the enderself and the sersed and as a the concertion of the series of the self in the self--and the serse and and the seried enes and seast and the sense and the eadure to the self and the present and as a to the self--and the seligious and the enders
    
    diversity 0.500000:
    can is reast to as a seligut and the complesed
    has fool which the self as it is a the beasing and us immery and seese for entoured underself of the seless and the sired a mears and everyther to out every sone thes and reapres and seralise as a streed liees of the serse to pease the cersess of the selung the elie one of the were as we and man one were perser has persines and conceity of all self-el
    
    diversity 1.000000:
    entoles by
    their lisevers de weltaale, arh pesylmered, and so jejurted count have foursies as is
    descinty iamo; to semplization refold, we dancey or theicks-welf--atolitious on his
    such which
    here
    oth idey of pire master, ie gerw their endwit in ids, is an trees constenved mase commars is leed mad decemshime to the mor the elige. the fedies (byun their ope wopperfitious--antile and the it as the f
    
    diversity 1.200000:
    cain, elvotidue, madehoublesily
    inselfy!--ie the rads incults of to prusely le]enfes patuateded:.--a coud--theiritibaior "nrallysengleswout peessparify oonsgoscess teemind thenry ansken suprerial mus, cigitioum: 4reas. whouph: who
    eved
    arn inneves to sya" natorne. hag open reals whicame oderedte,[fingo is
    zisternethta simalfule dereeg hesls lang-lyes thas quiin turjentimy; periaspedey tomm--whach 

.. [#rnn_reference] 此处的任务及实现参考了 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
