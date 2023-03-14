import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.utils import class_weight, shuffle

from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint 
import tensorflow as tf

def is_valid_folder(folder):
    return not folder.startswith('.')

def load_dataset(path, n_classes):
    foldernames = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    categories = []
    files = []
    for k, folder in enumerate(foldernames):
        filenames = os.listdir(path + folder);
        for file in filenames:
            files.append(path + folder + "/" + file)
            categories.append(k)

    df = pd.DataFrame({
        'filename': files,
        'category': categories
    })
    train_df = pd.DataFrame(columns=['filename', 'category'])
    for i in range(n_classes):
        train_df = train_df.append(df[df.category == i].iloc[:500,:])

    train_df = train_df.reset_index(drop=True)

    return train_df

def preprocess_images(train_df):
    def centering_image(img):
        size = [256,256]

        img_size = img.shape[:2]

        # centering
        row = (size[1] - img_size[0]) // 2
        col = (size[0] - img_size[1]) // 2
        resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
        resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

        return resized

    images = []
    with tqdm(total=len(train_df)) as pbar:
        for i, file_path in enumerate(train_df.filename.values):
            #read image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #resize
            if(img.shape[0] > img.shape[1]):
                tile_size = (int(img.shape[1]*256/img.shape[0]),256)
            else:
                tile_size = (256, int(img.shape[0]*256/img.shape[1]))

            #centering
            img = centering_image(cv2.resize(img, dsize=tile_size))

            #out put 224*224px 
            img = img[16:240, 16:240]
            images.append(img)
            pbar.update(1)

    images = np.array(images)

    return images

def split_data(train_df, images):
    y = train_df['category']
    x = train_df['filename']

    x, y = shuffle(x, y, random_state=8)

    data_num = len(y)
    random_index = np.random.permutation(data_num)

    x_shuffle = []
    y_shuffle = []
    for i in range(data_num):
        x_shuffle.append(images[random_index[i]])
        y_shuffle.append(y[random_index[i]])

    x = np.array(x_shuffle) 
    y = np.array(y_shuffle)
    val_split_num = int(round(0.2*len(y)))
    x_train = x[val_split_num:]
    y_train = y[val_split_num:]
    x_test = x[:val_split_num]
    y_test = y[:val_split_num]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def create_model(n_classes):
    img_rows, img_cols, img_channel = 224, 224, 3

    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(n_classes, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, x_test, y_test):
    batch_size = 32
    epochs = 60

    train_datagen = ImageDataGenerator(
            rotation_range=30, 
            width_shift_range=0.1,
            height_shift_range=0.1, 
            horizontal_flip=True)
    train_datagen.fit(x_train)

    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc')]
    )

    return history

def main():
    path = "datasets/drawing/"
    foldernames = [folder for folder in os.listdir(path) if is_valid_folder(folder)]
    n_classes = len(foldernames)

    train_df = load_dataset(path, n_classes)
    images = preprocess_images(train_df)
    x_train, y_train, x_test, y_test = split_data(train_df, images)

    model = create_model(n_classes)
    history = train_model(model, x_train, y_train, x_test, y_test)

    # Save the model
    with open('model2.tflite', 'wb') as f:
        f.write(tf.lite.TFLiteConverter.from_keras_model(model).convert())

if __name__ == "__main__":
    main()

    