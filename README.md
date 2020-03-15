# Word2Vec-LSTM中文文本分类
1. 环境    
python3   
tensorflow     
gensim     
jieba     
scipy     
numpy    
scikit-learn    

2. 数据集
数据集存放在data文件夹中，分为train.txt, test.txt, val.txt。   
数据集格式: 标签+'\t'+文本     
本实验为笔者自己搜集的论文数据集，有兴趣的读者请自行收集。    
该数据集设计六个类别：['摘要', '引言', '相关研究', '方法', '实验', '结语']    

3. 模型与数据预处理
lstm模型存放在rnn_model.py中，数据集的处理程序再loader.py中

4. 使用
(1)运行train_word2vec.py，对train.txt中数据使用jieba分词进行中文分词，后利用word2vec训练词向量     
(2)运行rnn_train.py，训练模型    
(3)运行rnn_test.py, 测试模型   
(4)运行rnn_predict.py, 进行模型预测

5. 参考
(1) https://github.com/gaussic/text-classification-cnn-rnn      
(2) https://github.com/cjymz886/text-cnn   
