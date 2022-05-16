import os
from datasets import load_dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
from util import log


# 定义在每条数据上执行的函数
def tokenize_function(example, max_len=256):
    return tokenizer(example["text"], max_length=max_len, truncation=True)


if __name__ == '__main__':
    logger = log.get_logger('imdb_bert')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用第几块GPU
    dataset_name = "imdb"  # huggingface上的数据集名称
    checkpoint = "bert-base-uncased"  # huggingface上的模型名称
    batch_size = 16
    num_epochs = 5
    path = '../trained_models/tmp'
    raw_datasets = load_dataset("imdb")
    logger.info("load data successfully!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    logger.info('load tokenizer successfully')
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    logger.info('load model successfully')
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 批量序列化
    logger.info('tokenize successfully')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")  # 在每一个batch动态padding
    tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(  # 训练集
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["label"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(  # 测试集
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["label"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    num_train_steps = len(tf_train_dataset) * num_epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
    )
    opt = Adam(learning_rate=lr_scheduler)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    logger.info('model compile successfully, start to train!')
    model.fit(tf_train_dataset, epochs=num_epochs)
    logger.info('train finished!')
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
    logger.info('model save successfully!')
