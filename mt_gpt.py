import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
import numpy as np
import random

from gpt import create_gpt_model


train_fn = "train-eng-spa.tsv"

train_pairs = []
with open(train_fn, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        train_pairs.append(line.split("\t"))


START_TOKEN = "[start]"
END_TOKEN = "[end]"
TO_TOKEN = "[to]"


def to_prompt(src):
    return START_TOKEN + " " + src + " " + TO_TOKEN + " "


def process_pair(p):
    return to_prompt(p[0]) + p[1] + " " + END_TOKEN


bitexts = list(map(process_pair, train_pairs))

for _ in range(5):
    print(random.choice(bitexts))


vocab_size = 16000  # Only consider the top 20k words
maxlen = 64  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

batch_size = 128

text_ds = tf.data.Dataset.from_tensor_slices(bitexts)
text_ds = text_ds.shuffle(buffer_size=4096)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return lowercase


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
print("Adapting data...")
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)


word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index


END_ID = word_to_index[END_TOKEN]


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"generated text:\n{txt}\n")


class Generator:
    def __init__(self, model, max_tokens, index_to_word, top_k=10):
        self.model = model
        self.max_tokens = max_tokens
        self.index_to_word = index_to_word
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def generate(self, start_tokens):
        start_tokens = [_ for _ in start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            if sample_token == END_ID:
                break
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([self.detokenize(_) for _ in tokens_generated])
        return txt


start_prompt = to_prompt("this is a book.")
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

model = create_gpt_model(embed_dim, num_heads, feed_forward_dim, vocab_size, maxlen)
print(model.summary())

print("Training...")

model.fit(text_ds, verbose=1, epochs=1, callbacks=[text_gen_callback])

weights_file = "gpt.h5"
model.save_weights(weights_file)

generator = Generator(model, 40, vocab)


def translate(src):
    start_prompt = to_prompt(src)
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    trans = generator.generate(start_tokens)

    return trans


def eval_file(tsv_fn):
    srcs = []
    refs = []
    with open(tsv_fn, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            pair = line.split("\t")
            srcs.append(pair[0])
            refs.append(pair[1])

    translations = []
    for src in srcs:
        trans = translate(src)
        translations.append(trans)

    import sacrebleu

    bleu = sacrebleu.corpus_bleu(translations, [refs])
    print(bleu.score)


eval_file("test-eng-spa.tsv")
