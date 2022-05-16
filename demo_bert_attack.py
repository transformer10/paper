from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForMaskedLM
import numpy as np
import nltk
import tensorflow_hub as hub
import pandas as pd


# 定义受害者模型
class VictimModel:
    def __init__(self, path):
        # 从本地加载分析器和模型（序列分类）
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(path)

    # 返回模型关于输入文本在各个标签上的logit
    def predict(self, text):
        tokens = self.tokenizer(text, return_tensors='tf')
        return self.model(**tokens).logits.numpy()


# 定义攻击模型
class AttackModel:
    def __init__(self, checkpoint):
        # 从huggingface上加载对应于该checkpoint的分词器和模型（MLM）
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = TFAutoModelForMaskedLM.from_pretrained(checkpoint)

    # 返回所有"[MASK]"位置上分数排名前topn的单词，shape为(掩码单词数, topn)
    def get_mask_words(self, text, topn):
        tokens = self.tokenizer(text, return_tensors='tf')
        pred_logits = self.model(**tokens).logits.numpy()
        ids = tokens['input_ids'][0].numpy()
        # 获取所有掩码位置的下标
        masked_indexes = np.where(ids == self.tokenizer.mask_token_id)[0]
        # 遍历所有掩码位置，并对于掩码位置上按照单词分数降序排序
        masked_ids_sorted = [np.argsort(-pred_logits[0][index])[:topn] for index in masked_indexes]
        mask_words = [self.tokenizer.convert_ids_to_tokens(id) for id in masked_ids_sorted]
        return mask_words


# 定义单词级别的分词器
class TokenModel:
    def __init__(self):
        pass

    # 将文本按照单词级别分词
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    # 将token列表还原成文本
    def detokenize(self, tokens):
        return nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(tokens=tokens)

    # 获取所有标点符号
    def get_punctuations(self):
        return ['.', ',', '?', '!']


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
            return np.zeros((1, 300))
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


if __name__ == '__main__':
    path = 'imdb_model'
    checkpoint = 'bert-base-uncased'
    text = 'The characters, cast in impossibly situation, [MASK] are totally estranged from reality.'
    victim_model = VictimModel(path)
    attack_model = AttackModel(checkpoint)
    print(attack_model.get_mask_words(text, 10))
