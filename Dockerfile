from python

RUN pip3 install gensim

CMD ["python3", "/root/project/word2vec/train_model.py", "/root/project/data/baike_line_participle.txt","/root/project/data/model/baike.model","/root/project/data/model/baike.vector"]