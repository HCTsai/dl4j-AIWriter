###Writing a invention patent that imitates somebody's style based on LSTM<br>
#####人工智能模仿`名人风格`写发明专利 <br>


![arch](https://raw.githubusercontent.com/HCTsai/dl4j-AIWriter/master/img/AIWriterArchitecture.png) 


####Example: Writing a patent like Jay Chou
#####范例：有 `周杰伦` 风范的一篇发明专利
```
    本发明公开 制冷模式状态下的眼神一片
    空气质量获取传感器设置有警报
    设置发射模型连接 控制模块用于
    空气调节端连接数据设置有数据线
    及室内环境又或为什么 妳在功能模块 接收
    主要模块包括信号输出端连接
    进行WIFI 雾化器的固定端连接
    本发明公开了一种红外线上
    设定车内能内部可吸入颗粒物和较梦去
    蠳蹲放申请至通过报警
    控制面板 PCB阈值 挂机设有别对
    管包括为什么的准备吃都可以
    保证音频低于第一风机电机调整
    发射器与数据处理模块与输出模块
    通过家庭数据处理模块用于电风扇方式
    自动依次及第二设定机组监测器
    雾化器的对流种大固定
    公开特征了 包括当前的关闭与插座
    电器信息进默默部分 车内的
    净化室内值本气体路由器和湿度
    获取采集运行灵魂里 觉得忘记 ...
```
#####Write again:
#####请 `"周杰伦"` 再写另一篇专利
```
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
```

####Usage ：

Step 1: Preprocess the training data ：

* 1.Word segmentation on Jay lyrics. (data/segres_jay.txt) <br>
* 2.Word segmentation on technical patents. (data/segres_patent_2675013w.txt )<br>
* 3.Clear special characters. <br>
* 4.Shuffle Sentences. (data/segres_patent_jay.txt)

Step 2: Train a model :

* TrainWordLSTM.java  : <br>
    Create Many-to-One Recurrent Neural Network by using dl4j.<br>

Step 3: Sample from the model :
* AIWordsWriter.java  : <br>
    Writing article based on the LSTM Model.<br>

####Requirements: 
- [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)


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

#### Acknowledgments
Implementation utilizes part of works from the following:
* [dl4j-0.4-examples](https://github.com/deeplearning4j/dl4j-0.4-examples)

