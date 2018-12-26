# 闲聊对话机器人（SEQ2SEQ）

序列到序列模型（Sequence to Sequence, SEQ2SEQ）是一种基于RNN的Encoder-Decoder结构，它也是现在谷歌应用于线上机器翻译的算法，翻译质量已经和人类水平不相上下。

这里的Encoder-Decoder结构，简单的来说就是算法包含两部分，一个负责对输入的信息进行Encoding，将输入转换为向量形式；然后由Decoder对这个向量进行解码，还原为输出序列。

关于SEQ2SEQ的原理，可以参考：

- To be added
- To be added

这里，我们使用SEQ2SEQ来实现一个闲聊（Chit Chat）对话机器人。除了闲聊机器人（输入一句话，输出一句回复）之外，它也可以被直接应用于解决其他类似问题，比如：翻译（输入一句英文，输出一句中文）、摘要（输入一篇文章，输出一份总结）、作诗（输入几个关键字，输出一首短诗）、对对联（输入上联，输出下联），等等。

这个任务对比与之前的RNN尼采风格文本生成，区别在于我们预测的不再是文本的连续字母概率分布，而是通过一个序列，来预测另外一个对应的完整序列。举例来说，针对一句常见的打招呼：

```text
How are you
```

这个句子（序列）一共有3个单词。当我们听到这个由3个单词组成的句子后，根据我们的习惯，我们最倾向与回复的一句话是"Fine thank you"。我们希望建立这样的模型，输入num_batch个由编码后单词和字符组成的，长为max_length的序列，输入张量形状为[num_batch, max_length]，输出与这个序列对应的序列（如聊天回复、翻译等）中单词和字符的概率分布，概率分布的维度为词汇表大小voc_size，输出张量形状为[num_batch, max_length, voc_size]。

首先，还是实现一个简单的 ``DataLoader`` 类来读取文本，

```py
DATASET_URL = 'https://github.com/huan/concise-chit-chat/releases/download/v0.0.1/dataset.txt.gz'
DATASET_FILE_NAME = 'concise-chit-chat-dataset.txt.gz'
LATENT_UNIT_NUM = 100
EMBEDDING_DIM = 50
MAX_LEN = 20
DONE = '\a'
GO = '\b'


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
        if len(text.split()) > MAX_LEN:
            text = (' ').join(text.split()[:MAX_LEN])
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

其次，我们还需要基于 `DataLoader` 加载的文本数据，建立一个词汇表 `Vocabulary` 来负责管理以下5项任务：

1. 将所有单词和标点符号进行编码；
2. 记录词汇表大小；
3. 建立单词到编码数字，以及编码数字到单词的映射字典；
4. 负责将文本句子转化为填充后的编码序列，形状为[batch_size, max_length]；

```py
class Vocabulary:
    def __init__(self, text):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts([GO, DONE] + re.split(r'[\s\t\n]', text))
        self.size = 1 + len(self.tokenizer.word_index.keys())

    def texts_to_padded_sequences(self, text_list):
        sequence_list = self.tokenizer.texts_to_sequences(text_list)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequence_list, maxlen=MAX_LEN, padding='post', truncating='post')
        return padded_sequences
```

接下来进行模型的实现。我们建立一个ChitChat模型，在 `__init__` 方法中我们将 `Vocabulary` 中的词汇到编码字典 `word_index` 和编码到词汇字典 `index_word` ，以及词汇量 `voc_size` 保存备用，实例化一个常用的 `Embedding` 单元，以及一个 `Encoder` 子模型和对应的 `Decoder` 子模型。子模型中需要使用Embedding单元、代表序列开始的GO字符编码，以及词汇表尺寸，我们通过构造参数传给它们。我们首先对序列进行Encoder操作，即将编码序列 `inputs` 变换为一个上下文向量 `context` ，然后再对其进行Decoder操作，得到输出编码序列张量，即作为模型的输出。变换后的序列张量形状为[num_batch, max_length, voc_size]。

`ChitChat` 模型具体实现如下：

```py
class ChitChat(tf.keras.Model):
    def __init__(self, vocabulary):
        super().__init__()
        self.word_index = vocabulary.tokenizer.word_index
        self.index_word = vocabulary.tokenizer.index_word
        self.voc_size = vocabulary.size

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.voc_size, output_dim=EMBEDDING_DIM, mask_zero=True)
        self.encoder = ChitEncoder(embedding=self.embedding)
        self.decoder = ChatDecoder(embedding=self.embedding,
            indice_go=self.word_index[GO], voc_size=self.voc_size)

    def call(self, inputs, teacher_forcing_targets=None, training=None):
        context = self.encoder(inputs)
        return self.decoder(inputs=context, training=training,
            teacher_forcing_targets=teacher_forcing_targets)
```

ChitEncoder子模型输入num_batch个由编码后单词和字符组成的，长为max_length的序列，输入张量形状为[num_batch, max_length]，输出与这个序列对应的上下文张量。为了简化代码，我们这里只使用一个最基本的LSTM单元，没有使用可以获得更佳效果的双向RNN、注意力机制等方法。在 `__init__` 方法中我们实例化一个常用的 `LSTM` 单元，并将其设置为 `return_state=True` 来获得最终的状态输出，我们首先对序列进行LSTM操作，即将编码序列变换为LSTM最终输出的状态 ，并将其作为代表编码序列的上下文信息 `context` ，作为模型的输出。

`ChitEncoder` 子模型具体实现如下：

```py
class ChitEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm_encoder = tf.keras.layers.LSTM(
          units=LATENT_UNIT_NUM, return_state=True)

    def call(self, inputs):
        _, *context = self.lstm_encoder(inputs)
        return context
```

ChatDecoder子模型输入num_batch个上下文信息张量 `context` 。在 `__init__` 方法中我们保存 `embedding` 单元，序列起始标志GO字符的编码 `indice_go` 和词汇表容量 `voc_size` ，实例化一个常用的 `LSTM` 单元，并将其设置为输出单元状态 `return_state=True` 来获得LSTM的状态输出，以及一个全连接层 `Dense` 单元，负责将LSTM的输出变换为最终的单词字符分布概率，并将其作为这个上下文信息对应的单词符号序列概率分布张量，作为模型的输出，形状为[num_batch, max_length, voc_size]。

`ChitEncoder` 子模型具体实现如下：

```py
class ChatDecoder(tf.keras.Model):
    def __init__(self, embedding, voc_size, indice_go):
        super().__init__()

        self.embedding = embedding
        self.indice_go = indice_go
        self.voc_size = voc_size

        self.lstm_decoder = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM, return_state=True)
        self.dense = tf.keras.layers.Dense(units=voc_size)

    def call(self, inputs, teacher_forcing_targets=None, training=False):
        batch_size = tf.shape(inputs[0])[0]
        batch_go_one_hot = tf.ones([batch_size, 1, 1]) \
            * [tf.one_hot(self.indice_go, self.voc_size)]

        outputs = tf.zeros([batch_size, 0, self.voc_size])
        output = batch_go_one_hot
        states = inputs

        for t in range(0, MAX_LEN):
            if training:
                target_indice = tf.expand_dims(
                    teacher_forcing_targets[:, t], axis=-1)
            else:
                target_indice = tf.argmax(output, axis=-1)

            decoder_inputs = self.embedding(target_indice)
            output, *states = self.lstm_decoder(
                inputs=decoder_inputs, initial_state=states)
            output = self.dense(output)
            outputs = tf.concat([outputs, output], 1)

        return outputs
```

训练过程与前节基本一致，在此复述：

- 从DataLoader中随机取一批训练数据；
- 将这批数据送入模型，计算出模型的预测值；
- 将模型预测值与真实值进行比较，计算损失函数（loss）；
- 计算损失函数关于模型变量的导数；
- 使用优化器更新模型参数以最小化损失函数。

```py
def loss(model, x, y) -> tf.Tensor:
    weights = tf.cast(tf.not_equal(y, 0), tf.float32)
    prediction = model(inputs=x, teacher_forcing_targets=y, training=True)
    return tf.contrib.seq2seq.sequence_loss(
        prediction, tf.convert_to_tensor(y), weights)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

learning_rate = 1e-2
num_batches = 8000
batch_size = 256

data_loader = DataLoader()
vocabulary = Vocabulary(data_loader.raw_text)
chitchat = ChitChat(vocabulary=vocabulary)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for batch_index in range(num_batches):
    queries, responses = data_loader.get_batch(batch_size)

    queries_sequences = vocabulary.texts_to_padded_sequences(queries)
    responses_sequences = vocabulary.texts_to_padded_sequences(responses)

    grads = grad(chitchat, queries_sequences, responses_sequences)
    optimizer.apply_gradients(grads_and_vars=zip(grads, chitchat.variables))

    print("step %d: loss %f" % (batch_index,
        loss(chitchat, queries_sequences, responses_sequences).numpy())
```

训练大约需要…… （时间、step数）

```sh
step 0: loss 8.019347
step 10: loss 5.006745
step 20: loss 4.798050
step 30: loss 4.728132
step 40: loss 3.921826
```

训练loss到小于…… 的时候，可以使用：

最后，我们需要一个用来对话的程序，来测试实际效果：

给 ChitChat 增加 predict 方法：

```py
class ChitChat(tf.keras.Model):
    # ... append the following code to previous code
    def predict(self, inputs, temperature=1.):
        inputs = np.expand_dims(inputs, 0)
        outputs = tf.squeeze(self(inputs))

        response_indices = []
        for t in range(1, MAX_LEN):
            output = outputs[t]
            indice = self.__logit_to_indice(output, temperature=temperature)
            if indice == self.word_index[DONE]:
                break
            response_indices.append(indice)
        return response_indices

    def __logit_to_indice(self, inputs, temperature=1.):
        inputs = tf.squeeze(inputs)
        prob = tf.nn.softmax(inputs / temperature).numpy()
        indice = np.random.choice(self.voc_size, p=prob)
        return indice
```

Chat 程序……

具体实现如下：

```py
data_loader = DataLoader()
vocabulary = Vocabulary(data_loader.raw_text)

chitchat = ChitChat(vocabulary)
checkpoint = tf.train.Checkpoint(model=chitchat)

checkpoint.restore(tf.train.latest_checkpoint('./checkpoints'))

index_word = vocabulary.tokenizer.index_word
word_index = vocabulary.tokenizer.word_index

while True:
    try:
        query = input('> ').lower()
        if query == 'q' or query == 'quit':
            break
        query = data_loader.preprocess(query)
        query = '{} {} {}'.format(GO, query, DONE)

        query_sequence = vocabulary.texts_to_padded_sequences([query])[0]
        response_sequence = chitchat.predict(query_sequence, temperature=0.5)

        response_word_list = [
            index_word[indice]
            for indice in response_sequence
            if indice != 0 and indice != word_index[DONE]
        ]

        print('Bot:', ' '.join(response_word_list))

    except KeyError:
        print("OOV: Please use simple words with the ChitChat Bot!")
```

生成的对话如下：

```shell
> hi
Bot:
> hello
Bot:
> faint
Bot:
```
