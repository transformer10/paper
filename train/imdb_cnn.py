import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import AutoTokenizer

from util import log


def encode(texts, tokenizer):
    data = []
    for text in texts:
        data += [tokenizer.encode(text)]
    return data


if __name__ == '__main__':
    logger = log.get_logger('imdb_bert')
    checkpoint = "bert-base-uncased"  # huggingface上的模型名称
    max_len = 400
    embedded_size = 300  # 每一个单词表示的维度数
    batch_size = 16
    num_epochs = 5
    raw_datasets = load_dataset("imdb")
    logger.info("load data successfully!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    shuffle_train_dataset = raw_datasets['train'].shuffle()
    tokenized_train_dataset = encode(shuffle_train_dataset['text'], tokenizer)
    tokenized_test_dataset = encode(raw_datasets['test']['text'], tokenizer)
    train_label = shuffle_train_dataset['label']
    test_label = raw_datasets['test']['label']
    X_train = tf.ragged.constant(tokenized_train_dataset)
    X_train = X_train.to_tensor(default_value=0, shape=[None, max_len])
    X_test = tf.ragged.constant(tokenized_test_dataset)
    X_test = X_test.to_tensor(default_value=0, shape=[None, max_len])
    num_train_steps = len(X_train) // batch_size * num_epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
    )
    opt = Adam(learning_rate=lr_scheduler)
    model = keras.models.Sequential([
        keras.layers.Embedding(30000, embedded_size, input_shape=[max_len]),
        keras.layers.Conv1D(filters=64, kernel_size=4, strides=2, padding="valid"),
        # keras.layers.Conv1D(filters=32, kernel_size=4, strides=2, padding="valid"),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, tf.constant(train_label), epochs=num_epochs, batch_size=batch_size,
              validation_data=(X_test, tf.constant(test_label)))
