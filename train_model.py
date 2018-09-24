import logging
import multiprocessing
import sys

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


def train(inp, outp1, outp2):
    # min_count小于这个次数的单词会被丢弃，默认为5
    # 大的size需要更多的训练数据, 但是效果会更好. 推荐值为几十到几百
    # workers并行度
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)


if __name__ == '__main__':
    dir_path = 'D:/project/datainsights/NLP/data/baike/data/'
    save_dir = dir_path + 'model/'
    inp = dir_path + 'baike_word_cn.txt'
    outp1 = dir_path + 'model/' + 'baike.cn.text.model'
    outp2 = dir_path + 'model/' + 'baike.cn.text.vector'

    # 若用户输入参数
    if len(sys.argv) == 4:
        inp = sys.argv[1]
        outp1 = sys.argv[2]
        outp2 = sys.argv[3]

    if not os.path.exists(os.path.dirname(outp1)):
        os.makedirs(os.path.dirname(outp1))

    logger.info('输入文件：' + inp)
    train(inp, outp1, outp2)
