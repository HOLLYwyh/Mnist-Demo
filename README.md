# “手写数字识别代”代码复现
1953608 吴英豪

[toc]

#  一、开发环境

开发语言 :  python 3.6.13

使用框架 ：TensorFlow 2.5.0  + Keras 2.5.0

开发工具 ：PyCharm 2020.2.2 x64

#  二、项目目录说明

此次复现的主要代码文件有：

1. simpleDemo.py

   参考了书目《Python深度学习》，是一个最简单的数字识别。

   从Kears的datasets中导入mnist, 并使用简单的隐藏层进行训练。

2. complexDemo.py

   添加多层神经网络的较复杂数字之别。

   从Keras的datasets中导入mnist,并使用带有均值池化的卷积神经网络方法训练。

3. finalDemo.py

   最终的数字识别。

   为了验证算法的可行性以及最终的正确性，这里采用了Kaggle比赛中Digit Recognizer题目所提供的的mnist测试集test.csv和训练集train.csv， 并采用带有最大池化的卷积神经网络，以及避免过拟合的方法进行训练。

4. input文件夹

   存放从finalDemo中读取的文件，训练集train.csv和测试集test.csv。

5. output文件夹

   存放finalDemo.py最终生成的csv文件。

# 三、代码与算法流程结果分析

首先需要将二维的数据一维化

![image-20210619212954584](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210619212954584.png)

1. simpleDemo.py      

   最简单的程序

   - 搭建神经网络

   ![image-20210619212745411](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210619212745411.png)

   - 算法分析

   （1）使用Sequential()函数作为模型。

   （2）建立一个有15个神经元，并采用relu函数作为激活函数的全连接层。**这里不使用其他函数的原因为防止梯度弥散。**

   （3）建立一个有10个神经元，并采用softmax函数作为激活函数的一个输入层。

2. complexDemo.py

   稍复杂的程序

   - 搭建神经网络

   ![image-20210619212639297](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210619212639297.png)

   - 算法分析

   使用卷积神经网络进行训练

   （1）添加三个卷积层

   （2）每个卷积层之间添加了一个均值池化。

   （3）添加一个Flatten层，用于由卷积层向全连接层过渡。

   （4）最后添加一个全连接层

3. finalDemo.py

   最终程序

   - 搭建神经网络

   ![image-20210619213442030](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210619213442030.png)

   - 算法分析

   同样使用卷积神经网络进行训练

   （1）添加四个卷积层

   （2）采用两个最大池化层

   （3）添加Dropout层防止过拟合

   （4）添加Flatten层，用于由卷积层向全连接层过渡。

   （5）添加了梯度下降优化器。

# 四、结果分析

```python
epochs=50
```

三个Demo均采用50个回合进行训练

 ```python
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
 ```

采用network.evaluate()评估函数输出  测试集损失和测试集准确率，训练集损失以及训练集准确率。

1. simpleDemo.py

   ![image-20210621082733831](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210621082733831.png)

   可以发现simpleDemo的预测准确性并不是很高，仅有92%左右。

2. complexDemo.py

   ![image-20210621082848033](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210621082848033.png)

   预测准确性可以达到99%左右，但是仍然存在着过拟合的情况。

3. finalDemo.py

   ![image-20210621083247992](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210621083247992.png)

   在Kaggle提供的测试集和训练集上训练效果较好，测试集上的准确性能够达到99.55%左右，最终在Kaggle上的效果也证明了此算法较好。

   ![image-20210621083448474](C:\Users\吴英豪\AppData\Roaming\Typora\typora-user-images\image-20210621083448474.png)