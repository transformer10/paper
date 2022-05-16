"""
    transformers加载bert模型和分词器
    tensorflow_hub加载use，将句子编码为定长向量
    nltk用来分词，词性标注等
    numpy进行向量、矩阵操作
    pandas加载训练好的词向量（counter-fitted-vectors）
"""

import nltk
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from util import log


# 定义被攻击的模型
class VictimModel:
    # 根据路径初始化分词器和模型
    def __init__(self, path):
        self.text = None
        self.encoding = None
        self.score = None
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(path)
        self.stop_words = ['.', ',', '?', '!'] + nltk.corpus.stopwords.words('english')

    # 根据编码对文本分词，以及得到不在停用词表中的单词的id
    def __set_token_ids(self):
        word_nums = self.encoding.word_ids()[-2] + 1
        self.ids = []
        self.words = []
        for i in range(word_nums):
            start, end = self.encoding.word_to_chars(i)
            word = (self.text[start:end]).lower()
            self.words.append(word)
            if word not in self.stop_words:
                self.ids.append(i)

    # 根据单词id，得到模型去掉该id后的分数
    def __set_id_scores_all_labels(self):
        self.id_to_score_all_label = dict()
        self.id_to_score = dict()
        true_label = np.argmax(self.score)
        for i in self.ids:
            start, end = self.encoding.word_to_chars(i)
            new_text = self.text[:start] + self.text[end:]
            token = self.tokenizer(new_text, return_tensors='tf')
            self.id_to_score_all_label[i] = self.model(**token).logits.numpy()[0]
            self.id_to_score[i] = self.score[true_label] - self.id_to_score_all_label[i][true_label]
            if self.id_to_score_all_label[i][true_label] < self.id_to_score_all_label[i][1 - true_label]:
                self.id_to_score[i] += (self.id_to_score_all_label[i][1 - true_label] - self.score[1 - true_label])
        keys = list(self.id_to_score.keys())
        values = list(self.id_to_score.values())
        index = np.argsort(values)
        self.ids = []
        for i in index[::-1]:
            self.ids.append(keys[i])

    # 根据输入文本，设置编码、token、以及id
    def set_text(self, text):
        self.text = text
        self.encoding = self.tokenizer(text, return_tensors='tf')
        self.score = self.model(**self.encoding).logits.numpy()[0]
        self.__set_token_ids()
        self.__set_id_scores_all_labels()

    # 模型在各个label上的分数
    def get_model_score(self):
        return self.score

    def get_words(self):
        return self.words

    def get_word_by_id(self, i):
        return self.words[i]

    def get_ids(self):
        return self.ids

    def get_id_scores_all_labels(self):
        return self.id_to_score_all_label

    def get_id_scores(self):
        return self.id_to_score

    def predict(self, text):
        t = self.tokenizer(text, return_tensors='tf')
        return self.model(**t).logits.numpy()[0]

    def test_model(self, dataset):
        def tokenize_function(example):
            return self.tokenizer(example["text"], max_length=256, truncation=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tf_dataset = tokenized_dataset.to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=16,
        )
        pred = self.model.predict(tf_dataset).logits
        class_pred = np.argmax(pred, axis=1)
        result = (class_pred == np.array(dataset['label']))
        print("the accuracy rate: %f", result.sum() / len(class_pred))


# 定义嵌入模型（词嵌入和文本嵌入）
class EmbeddingModel:
    def __init__(self, path='./word_vectors/counter-fitted-vectors.txt'):
        # 从本地加载所有单词的词嵌入向量
        self.word_embeddings = pd.read_csv(path, header=None, sep=' ')
        # 加载USE模型
        self.sentence_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # 返回文本的嵌入向量
    def get_sentence_embedding(self, text):
        return np.array(self.sentence_encoder([text]))[0]

    # 返回单词的嵌入向量
    def get_word_embedding(self, word):
        row = self.word_embeddings[self.word_embeddings[0] == word]
        if row.empty:
            return np.zeros(300)
        else:
            return np.array(row.iloc[0, 1:])

    # 计算两段文本的cos相似度
    def compare_sentences(self, text1, text2):
        sentence_embedding1 = self.get_sentence_embedding(text1)
        sentence_embedding2 = self.get_sentence_embedding(text2)
        return np.dot(sentence_embedding1, sentence_embedding2)

    # 计算两个单词的cos相似度
    def compare_words(self, word1, word2):
        word_embedding1 = self.get_word_embedding(word1)
        word_embedding2 = self.get_word_embedding(word2)
        return np.dot(word_embedding1, word_embedding2)

    # 获取一个单词相似度排在前k的单词
    def get_topk_similarity(self, word, k):
        embedding = self.get_word_embedding(word)
        simi = []
        for i in range(len(self.word_embeddings)):
            e = self.word_embeddings.iloc[i][1:]
            simi.append(np.dot(embedding, e))
        simi = np.array(simi)
        index = np.argsort(-simi)
        words = []
        for i in range(1, k + 1):
            words.append(self.word_embeddings[0][index[i]])
        return words


if __name__ == "__main__":
    logger = log.get_logger('demo_textfooler')
    text = "This movie is good."  # 被攻击的文本
    path = "./trained_models/imdb"  # 被攻击模型路径
    N = 10  # 对于每个位置取的同义词个数
    epsilon = 0.7  # 句向量的相似度阈值，大于等于该值才被加入候选列表
    victim_model = VictimModel(path)
    embedding_model = EmbeddingModel()
    word_detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()  # 将token序列转换成为文本
    victim_model.set_text(text)
    words = victim_model.get_words()
    ids = victim_model.get_ids()
    tags = nltk.pos_tag(words)
    score_all_label = victim_model.get_model_score()
    true_label = np.argmax(score_all_label)
    true_score = score_all_label[true_label]
    text_adv = text
    tmp_score = true_score
    logger.debug("the information of origin text: ")
    logger.info("the origin text: %s", text)
    logger.info("the tokens of origin text: %s", words)
    logger.info("the true label of origin text: %s, the corresponding score: %s", true_label, true_score)
    logger.info("the position can replace: %s", ids)
    logger.info("the corresponding words of the position: %s", np.array(words)[ids])
    logger.debug("\nbegin to attack:")
    for cnt, i in enumerate(ids):  # 替换第i个单词
        word = words[i]
        tag = tags[i][1]
        similar_words = embedding_model.get_topk_similarity(word, N)
        logger.info("attack number: %d", cnt + 1)
        logger.info("current text: %s", text_adv)
        logger.info("current score: %f", tmp_score)
        logger.info("the attack origin word: %s", word)
        logger.info("the part of speech of the word: %s", tag)
        logger.info("the top %d similar words: %s", N, similar_words)
        candidate = []
        fin_candidate = []
        y = []  # 第i个单词替换后的标签
        p = []  # 第i个单词替换后相应标签的分数
        s = []  # 第i个单词替换后和原始文本的相似度
        for new_word in similar_words:  # 先根据词性对第i个单词的替换列表进行过滤
            words[i] = new_word
            new_tag = nltk.pos_tag(words)[i][1]
            if tag == new_tag:
                candidate.append(new_word)
        logger.info("the similar words have the same pos as the original word: %s", candidate)
        for new_word in candidate:
            words[i] = new_word
            new_text = word_detokenizer.detokenize(tokens=words)
            sim = embedding_model.compare_sentences(text_adv, new_text)
            if sim >= epsilon:
                score = victim_model.predict(new_text)
                label = np.argmax(score)
                fin_candidate.append(new_word)
                y.append(label)
                p.append(score[label])
                s.append(sim)
        logger.info("replace words with a similarity greater than %.2f: %s", epsilon, fin_candidate)
        logger.info("the corresponding label: %s", y)
        logger.info("the corresponding score: %s", p)
        logger.info("the corresponding sentence similarity: %s", s)
        index = np.array(y) != true_label
        if np.sum(index) > 0:
            y_valid = np.where(np.array(y) != true_label)[0]
            s_valid = np.array(s)[index]
            words[i] = fin_candidate[y_valid[np.argsort(s_valid)[-1]]]
            text_adv = word_detokenizer.detokenize(tokens=words)
            logger.info("\nsuccessfully! The adversarial text: %s", text_adv)
            break
        else:
            min_p = np.min(p)
            if min_p < tmp_score:
                tmp_score = min_p
                words[i] = fin_candidate[np.argsort(p)]
        logger.info("\n")
