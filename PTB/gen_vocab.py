import codecs
import collections
from operator import itemgetter

MODE = "TRANSLATE_ZH"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "./PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "../seq2seq/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "../seq2seq/TED_data/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000

# 对单词按照词频排序
counter = collections.Counter()
with codecs.open(RAW_DATA,"r","utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 插入特殊符号
if MODE == 'PTB':
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]

# 保存词汇表
with codecs.open(VOCAB_OUTPUT, "w", "utf-8") as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")
