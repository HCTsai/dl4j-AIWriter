###Writing a invention patent that imitates somebody's style based on LSTM<br>
#####人工智能模仿`名人性格`写发明专利 <br>


![arch](https://github.com/HCTsai/dl4j-AIWriter/blob/master/img/AIWriterArchitecture.png) 


####Example: Writing a patent by Jay style
#####范例：模仿 `周杰伦` 风格写发明专利:

    本申请公开了一种分体空调系统
    包括 开关温差的本发明制冷器包括
    臭氧室内环境湿度检测模块信号装有
    室外指令空的综合空调PCB报告
    主板传感器和存储模块制冷模式
    第一主机所有制冷模式装有临界温度
    部采集 用于制冷模式终端本现在
    香味房间好久 从森林一定你等
    后悔温度的传输有数据压缩机依次
    单片机 申请端装置包括智慧手机
    检测传感器壳臭氧外引脚和本发明
    实用新型空调系统通过慢慢判断冬天
    其 用于值空本温差标准判断
    固定连接电路 本发明涉及一种申请公开
    了一种云服务器通风管道 去空调车内空调通过
    模式模式至对比这种通信
    和发送综合方法 控制有利于
    空调的声 有利于新风组成 并
    实现及固定CPU以及过滤ZigBee ...
    
####Implementation ：

Preparing a training corpus：

* 1.Word segmentation on Jay lyrics. (data/segres_jay.txt)<br>
* 2.Word segmentation on technical patents. (data/segres_patent_2675013w.txt)<br>
* 3.Clearing special characters.<br>
* 4.Shuffle Sentences. (data/segres_patent_jay.txt)

Model Training :

* Create Many-to-One Recurrent Neural Network by using dl4j.
* Input:  Previous N words.
* Output: Next word prediciton.


####Code:

TrainWordLSTM.java  : <br>
>Traning a LSTM Model. <br>

AIWordsWriter.java  : <br>
>Writing article based on the LSTM Model.<br>

####Requirements: 
>Deeplearning4j(dl4j)<br>
>JBlas

####Materials:

<p>A series of RNN Tutorial:</p>
<!--
![arch](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png) 
-->
<ol>
<li><a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a></li>
<li><a href="http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/">Introduction to RNNs</a></li>
<li><a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a></li>
</ol>




