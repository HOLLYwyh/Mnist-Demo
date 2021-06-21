from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 搭建LeNet网络
def lenet():
    networks = models.Sequential()
    networks.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    networks.add(layers.AveragePooling2D((2, 2)))
    networks.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    networks.add(layers.AveragePooling2D((2, 2)))
    networks.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    networks.add(layers.Flatten())
    networks.add(layers.Dense(84, activation='relu'))
    networks.add(layers.Dense(10, activation='softmax'))
    return networks


network = lenet()
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=50, batch_size=128, verbose=2)
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
