import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam,Nadam, SGD

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
plt.style.use("ggplot")
#%matplotlib inline

import tensorflow as tf
print(tf.__version__)


#数据探索
#使用 CelebA 数据集，其中包括 178 x 218 像素的图像。
main_folder = "/home/wh/PycharmProjects/pythonProject/venv/毕业论文/"
images_folder = main_folder + "img_align_celeba/"

training_sample = 10000
validation_sample = 2000
test_sample = 2000
img_width = 178
img_height = 218
batch_size = 16

#加载每张图片的属性
df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True)

df_attr.head(5)
df_attr.describe()
df_attr.columns
df_attr.isnull().sum()
df_attr.shape
for i,j in enumerate(df_attr.columns):
    print(i+1, j)

#随机选取5张笑脸及5张非笑脸图像并可视化
Sm_img = df_attr[df_attr['Smiling'] == 1].sample(5)
no_Sm_img = df_attr[df_attr['Smiling'] == 0].sample(5)
plt.figure(figsize=(20, 10))
plt.suptitle("Smiling:",x=0.1,y=0.75, fontsize=15)
i = 0
for batch in list(Sm_img.index):
    bat = load_img(images_folder + batch)
    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(bat)

    if i == 4:
        break
    i = i + 1
plt.show()
i=5
for batch in list(no_Sm_img.index):
    bat = load_img(images_folder + batch)
    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(bat)

    if i == 9:
        break
    i = i + 1
plt.show()
plt.title("No Smiling:",x=-5.3,y=0.55, fontsize=15)


#属性分布
sns.countplot(df_attr["Smiling"])
plt.show()

#将数据集拆分为训练、验证和测试
#1-162770 是训练 162771-182637 是验证 182638-202599 是测试
#由于执行时间,现在我们将使用减少数量的图像：
#训练 20000 张图片 验证 5000 张图片 测试 5000 张图片
df_partition = pd.read_csv(main_folder + "list_eval_partition.csv")
df_partition.head(5)
df_partition.sample(100)
#0 =====> training  #1 =====> validation  #2 =====> testing
df_partition["partition"].value_counts().sort_index()

#将分区和属性连接在同一数据框中
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr["Smiling"], how="inner")
df_par_attr.head(5)

#生成分区（训练、验证、测试）
#为了使模型获得良好的性能，需要平衡图像数量，每个模型都有自己的训练、验证和测试平衡数据文件夹。
#创建有助于我们创建每个分区的函数。
def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,)+x.shape)
    return x


def generate_df(partition, attr, num_samples):
    df_ = df_par_attr[(df_par_attr['partition'] == partition)
                      & (df_par_attr[attr] == 0)].sample(int(num_samples / 2))
    df_ = pd.concat([df_,
                     df_par_attr[(df_par_attr['partition'] == partition)
                                 & (df_par_attr[attr] == 1)].sample(int(num_samples / 2))])
    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr], 2)
    # for Test
    else:
        x_ = []
        y_ = []
        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis=0)
            x_.append(im)
            y_.append(target[attr])
    return x_, y_

#预处理图像：数据增强
#为图像生成数据增强。
#数据增强允许生成对原始图像进行修改的图像。 该模型将从这些变化（改变角度、大小和位置）中学习，能够更好地预测在位置、大小和位置上可能具有相同变化的从未见过的图像。
#这就是图像在数据增强后的样子（基于下面给出的参数）。
# 生成用于数据增强的图像生成器
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True)
# 加载一张图片并重塑
# load a example image
example_pic = images_folder + "005040.jpg"
img = load_img(example_pic)
plt.grid(False)
plt.axis('off')
plt.imshow(img)
df_attr.loc[example_pic.split('/')[-1]][['Smiling','Male',"Young"]]

img = load_img(example_pic)
x = img_to_array(img) / 255.
x = x.reshape((1,) + x.shape)

# 绘制加载图像的 10 个增强图像
plt.figure(figsize=(20, 10))
plt.suptitle("Data augmentation", fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(batch.reshape(218, 178, 3))

    if i == 9:
        break
    i = i + 1

plt.show()
#结果是一组对原始图像进行了修改的新图像，允许模型从这些变化中学习，以便在学习过程中获取此类图像并预测更好的从未见过的图像。


# 构建数据生成器
# train data
x_train, y_train = generate_df(0, "Smiling", training_sample)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
# validation data
x_valid, y_valid = generate_df(1, "Smiling", validation_sample)
# test data
x_test, y_test = generate_df(2, 'Smiling', test_sample)

#定义模型结构。输出层将有2个神经元（等于类型的数量），我们将使用sigmoid作为激活函数。
#我将使用某一结构来解决这个问题。 也可以通过更改隐藏层数，激活函数和其他超参数来修改此架构。
from keras.layers import Conv2D, MaxPooling2D
cnnmodel = Sequential()
cnnmodel.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(218,178,3)))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(128, activation='relu'))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Dense(64, activation='relu'))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Dense(2, activation='sigmoid'))
cnnmodel.summary()
#编译模型。 使用binary_crossentropy作为损失函数，使用ADAM作为优化器：
cnnmodel.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#cnnmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# load the model
# redefine model to output right after the first hidden layer
ixs = [1, 3, 5, 7, 9]
outputs = [cnnmodel.layers[i].output for i in ixs]
model = Model(inputs=cnnmodel.inputs, outputs=outputs)
# load the image with the required shape
img = load_img(example_pic)
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 16
for fmap in feature_maps:
    ix = 1
    plt.figure(figsize=(8, 2))
    for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(2, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix - 1], cmap='viridis')
            ix += 1
    # show the figure
    plt.show()


#训练模型,并传入我们之前创建的验证数据，以验证模型的性能：
#cnnhist = cnnmodel.fit(train_generator, epochs=5, validation_data=(x_valid, y_valid), batch_size=4)
#cnnhist = cnnmodel.fit(train_generator, epochs=10, validation_data=(x_valid, y_valid), batch_size=4)
cnnhist = cnnmodel.fit(train_generator, epochs=20, validation_data=(x_valid, y_valid), batch_size=4)
# Plot loss function value through epochs
plt.figure(figsize=(18, 4))
plt.plot(cnnhist.history['loss'], label = 'train')
plt.plot(cnnhist.history['val_loss'], label = 'valid')
plt.legend()
plt.title('Loss Function')
plt.show()

# Plot accuracy through epochs
plt.figure(figsize=(18, 4))
plt.plot(cnnhist.history['accuracy'], label = 'train')
plt.plot(cnnhist.history['val_accuracy'], label = 'valid')
plt.legend()
plt.title('Accuracy')
plt.show()

# 生成预测
cnnmodel_prediction = [np.argmax(cnnmodel.predict(feature)) for feature in x_test]

# 报告测试准确度
test_accuracy = 100 * (np.sum(np.array(cnnmodel_prediction)==y_test)/len(cnnmodel_prediction))
print('model evaluation')
print("test accuracy : ", test_accuracy)
print('f1 score : ', f1_score(y_test, cnnmodel_prediction))




##import inceptionv3 model
inc_model = InceptionV3(weights="/home/wh/PycharmProjects/pythonProject/venv/毕业论文/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(img_height,img_width,3))

print("number of layers in the model : ", len(inc_model.layers))


#不包括顶层（包括分类）。这些图层将被以下图层替换：
#添加自定义图层
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# creating the final model
model_ = Model(inputs=inc_model.input, outputs=predictions)

# lock initial layers to not to be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#model_.summary()

# train the model
checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5', verbose=1, save_best_only=True)
hist = model_.fit_generator(train_generator, validation_data=(x_valid, y_valid), steps_per_epoch=training_sample/batch_size, epochs=20, callbacks=[checkpointer], verbose=1)


# plot loss with epochs
plt.figure(figsize=(18,4))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.legend()
plt.title('loss function')
plt.show()

# Plot accuracy through epochs
plt.figure(figsize=(18, 4))
plt.plot(hist.history['accuracy'], label = 'train')
plt.plot(hist.history['val_accuracy'], label = 'valid')
plt.legend()
plt.title('Accuracy')
plt.show()

# 加载最佳模型
model_.load_weights('weights.best.inc.male.hdf5')

# generate predictions
model_prediction = [np.argmax(model_.predict(feature)) for feature in x_test]

# 报告测试准确度
test_accuracy = 100 * (np.sum(np.array(model_prediction)==y_test)/len(model_prediction))
print('model evaluation')
print("test accuracy : ", test_accuracy)
print('f1 score : ', f1_score(y_test, model_prediction))


##vgg16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import adam_v2, rmsprop_v2
from keras.callbacks import EarlyStopping

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
# load base model
vgg16_weight_path = "/home/wh/PycharmProjects/pythonProject/venv/毕业论文/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
IMG_SIZE = (218,178,3)
vgg = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE
)

NUM_CLASSES = 2
#添加自定义图层
vgg16 = Sequential()
vgg16.add(vgg)
vgg16.add(layers.Dropout(0.3))
vgg16.add(layers.Flatten())
vgg16.add(layers.Dropout(0.5))
vgg16.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))
vgg16.layers[0].trainable = False

vgg16.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
#vgg16.summary()
vgg16_history = vgg16.fit_generator(train_generator,steps_per_epoch=training_sample/batch_size,epochs=5,validation_data=(x_valid, y_valid),verbose=1)

# plot loss with epochs
plt.figure(figsize=(18,4))
plt.plot(vgg16_history.history['loss'], label='train')
plt.plot(vgg16_history.history['val_loss'], label='validation')
plt.legend()
plt.title('loss function')
plt.show()

# Plot accuracy through epochs
plt.figure(figsize=(18, 4))
plt.plot(vgg16_history.history['accuracy'], label = 'train')
plt.plot(vgg16_history.history['val_accuracy'], label = 'valid')
plt.legend()
plt.title('Accuracy')
plt.show()

# generate predictions
vgg16_prediction = [np.argmax(vgg16.predict(feature)) for feature in x_test]

# 报告测试准确度
vgg16_test_accuracy = 100 * (np.sum(np.array(vgg16_prediction)==y_test)/len(vgg16_prediction))
print('model evaluation')
print("test accuracy : ", vgg16_test_accuracy)
print('f1 score : ', f1_score(y_test, vgg16_prediction))


##ensemble modle
ensemble = list(range(0,2000))
for i in list(range(0,2000)):
    if cnnmodel_prediction[i]==model_prediction[i]:
        ensemble[i]=cnnmodel_prediction[i]
    else:
        if model_prediction[i]==vgg16_prediction[i]:
            ensemble[i] = model_prediction[i]
        else:
            ensemble[i] = vgg16_prediction[i]

# 报告测试准确度
ensemble_test_accuracy = 100 * (np.sum(np.array(ensemble)==y_test)/len(ensemble))
print('model evaluation')
print("test accuracy : ", ensemble_test_accuracy)
print('f1 score : ', f1_score(y_test, ensemble))


# dictionary to name the prediction
def gender_prediction(filename):
    '''
    predict the Smiling
    input:filename: str of the file name
    return:array of the prob of the targets.
    '''
    im = cv2.imread(filename)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)
    # prediction
    result1 = cnnmodel.predict(im)
    result2 = model_.predict(im)
    result3 = vgg16.predict(im)
    if np.argmax(result1) == np.argmax(result2):
        prediction = np.argmax(result1)
    else:
        if np.argmax(result2) == np.argmax(result3):
            prediction = np.argmax(result2)
        else:
            prediction = np.argmax(result3)
    return prediction


# select random images of the test partition
df_to_test = df_par_attr[(df_par_attr['partition'] == 2)].sample(11)

plt.figure(figsize=(20, 10))
i = 0
for index, target in df_to_test.iterrows():
    prediction = gender_prediction(images_folder + index)
    # display prediction
    display_result(images_folder + index, prediction)
    filename=images_folder + index
    if i == 10:
        plt.suptitle('', fontsize=28)
        break
    if prediction > 0.5:
        result = 'Smile'
    else:
        result = 'Unsmile'
    img = load_img(filename)
    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.title(result, fontsize=24)
    plt.imshow(img)
    plt.axis('off')
    i = i + 1
plt.show()