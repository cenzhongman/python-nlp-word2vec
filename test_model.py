import gensim

module_path = 'D:/project/datainsights/NLP/data/source/baike/data/model/baike.cn.text.model'
model = gensim.models.word2vec.Word2Vec.load(module_path)
rst_list = model.most_similar("美国")
print(rst_list)
rst_list = model.most_similar("中国")
print(rst_list)