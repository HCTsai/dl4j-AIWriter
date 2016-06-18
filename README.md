###人工智能模仿`名人性格`写发明专利 <br>
#####Writing a invention patent that imitates somebody's style based on RNN with LSTM<br>


![arch](https://github.com/HCTsai/dl4j-AIWriter/blob/master/img/AIWriterArchitecture.png) 


####范例：模仿 `周杰伦` 写发明专利:
#####Example: Writing a patent by Jay style

    温柔本人体是的手机设定进行的 
    调节对比从方法 冷一电路板 智能 
    中央空调方法 发送智能人体湿度   
    输出控制风管 空调人体假装数据处理用户
    送风口设有控制器最佳实现浓度端方式 
    设置装置 交换器水槽浓度控制或者仔细
    到的体才过滤听装有
    落妳温度 开启装置发射的用于  
    终端带无线控制装置 空调空气质量核心 
    单元主板过传感器传感器应急信号
    该控制器 本发明根据为何保障舒适  
    启用心停留 相应的负离子达到致命伤
    可系统 步骤 其特征在于
    上通过累了一种管道控制慢慢实现
    信息通过计算闹钟的oh室外的
    方式 智能控制器特征在于 控制器室内发出  
    相连 想带 ...(略)

####实施方法(Implementation)：

语料准备(Prerequisite)：

* 1.将周杰伦歌词进行分词(Word Segmentation)。<br>
* 2.将专利样本文件进行分词。<br>
* 3.将周杰伦歌词与专利文件清洗掉特殊标点符号。<br>
* 4.以句子为单位交错(Shuffle)成一份新的语料，使模型可学习到两种不同语料的概念。(参考:data/segres_patent_jay.txt)<br>
   
模型训练(Model Training):

* 使用dl4j ，建立 Many-to-One Recurrent Neural Network 。从前面N个词，预测下一个词的输出。

文章产生:

* 将句子输入模型，产生下一个输出词的机率向量。<br>
* 若机率向量有显著高于平均机率的词，直接输出高机率的词。（使文章稳定理性）<br>
* 若机率向量中，每个词的输出机率都差不多，则输出随机范围中，最高机率的词。(使文章随机浪漫)<br>


####重要代码说明:

TrainWordLSTM.java  : <br>
>此代码会训练一个 LSTM Model ，输入为 Word Sequence，输出为下一个可能出现的文字。<br>

AIWordsWriter.java  : <br>
>此代码会载入训练好的 LSTM Model，输入一个起始词(Word)，并输出文章内容。<br>

####Requirements: 
>Deeplearning4j(dl4j)<br>
>JBlas

####Materials:

<p>A series of RNN Tutorial:</p>
![arch](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png) 
<ol>
<li><a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a></li>
<li><a href="http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/">Introduction to RNNs</a></li>
<li><a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a></li>
</ol>




