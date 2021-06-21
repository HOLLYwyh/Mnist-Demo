from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist

# 使用api加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28)).astype('float')  # 将二维转化为一维，数据的格式为float
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)  # one hot 编码，需要额外研究

# 搭建神经网络
network = models.Sequential()
network.add(layers.Dense(units=15, activation='relu', input_shape=(28*28, ),))
network.add(layers.Dense(units=10, activation='softmax'))

# 编译步骤
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=50, batch_size=128, verbose=2)

# 来在测试集上测试一下模型的性能吧
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)












