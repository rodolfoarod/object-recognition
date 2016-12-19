import sys
import argparse
import matplotlib.pyplot as plt
#import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras import backend as K

def main():
    """Main Function"""

    # Usage: obj_rec.py [-h] [-l FILENAME] [-e]
    parser = argparse.ArgumentParser(
        description='Object Recognition - Deep Learning')
    parser.add_argument(
        '-l', '--load', type=str, help='load model file', \
        metavar='FILENAME')
    parser.add_argument(
        '-e', '--evaluate', help='evaluate model with test images', \
        action='store_true')
    args = parser.parse_args()

    # load data set
    x_train, y_train, x_test, y_test, input_shape = load_cifar10_dataset()

    if args.load:
        print "\nLoading Model"
        model = load_model(args.load)
        model.summary()
        print "Model successfully loaded"

    if args.evaluate:
        score = model.evaluate(x_test, y_test)
        print "\nTest accuracy: %0.05f" % score[1]
        sys.exit()

    if not args.load:

        # create model
        nb_filters = 32
        pool_size = (2, 2)
        kernel_size = (3, 3)

        # the sequential model is a linear stack of layers
        model = Sequential()

        # Convolution operator for filtering windows of two-dimensional inputs
        model.add(Convolution2D(
            nb_filters, \
            kernel_size[0], \
            kernel_size[1], \
            border_mode='valid', \
            input_shape=input_shape))

        # Activation layer with relu
        model.add(Activation('relu'))

        model.add(Convolution2D(
            nb_filters, \
            kernel_size[0], \
            kernel_size[1]))

        model.add(Activation('relu'))

        # Max Pooling
        model.add(MaxPooling2D(pool_size=pool_size))

        # Dropout consists in randomly setting a fraction p
        # of input units to 0 at each update during training time,
        # which helps prevent overfitting.
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        # Flattens the input. Does not affect the batch size.
        model.add(Flatten())

        # Regular fully connected NN layer
        model.add(Dense(512))

        model.add(Activation('relu'))

        model.add(Dropout(0.5))

        model.add(Dense(10))

        model.add(Activation('softmax'))

        # model summary
        model.summary()

        # Configures the learning process
        model.compile(
            loss='categorical_crossentropy', \
            optimizer='adadelta', \
            metrics=['accuracy'])

    # train the model
    history = model.fit(
        x_train, \
        y_train, \
        batch_size=32, \
        nb_epoch=5, \
        verbose=1, \
        validation_split=0.1)

    # test the model
    score = model.evaluate(x_test, y_test)
    print "\nTest accuracy: %0.05f" % score[1]

    # Graph with training accuracy
    plot_training_history(history)

    # Saves model in a HDF5 file, includes:
    # Architecture of the model
    # Weights of the model
    # Training config
    # State of the optimizer
    model.save('obj_rec_model.h5')

    # Predict test images classes
    # img_classes = (
    #     "airplane", \
    #     "automobile", \
    #     "bird", \
    #     "cat", \
    #     "deer", \
    #     "dog", \
    #     "frog", \
    #     "horse", \
    #     "ship", \
    #     "truck")

    # y_hat = model.predict_classes(x_test)
    # y_test_array = y_test.argmax(1)
    # pd.crosstab(y_hat, y_test_array)
    # test_wrong = [im for im in zip(x_test, y_hat, y_test_array) if im[1] != im[2]]
    # plt.figure(figsize=(15, 15))
    # for ind, val in enumerate(test_wrong[:20]):
    #     plt.subplot(10, 10, ind + 1)
    #     im = val[0]
    #     plt.axis("off")
    #     plt.text(0, 0, img_classes[val[2]], fontsize=14, color='green') # correct
    #     plt.text(0, 32, img_classes[val[1]], fontsize=14, color='red')  # predicted
    #     plt.imshow(im, cmap='gray')
    # plt.show()


def load_cifar10_dataset():
    """Loads CIFAR10 data set"""
    from keras.datasets import cifar10

    ## 32x32 color images: shape -> (nb_samples, 3, 32, 32)
    img_w = img_h = 32
    img_channels = 3

    ## Load cifar10 data set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    ## th -> (samples, channels, height, width)
    if K.image_dim_ordering() == 'th':
        x_train = x_train.reshape(
            x_train.shape[0], img_channels, img_w, img_h)
        x_test = x_test.reshape(
            x_test.shape[0], img_channels, img_w, img_h)
        input_shape = (
            img_channels, img_w, img_h)
    ## tf -> (samples, height, width, channels)
    else:
        x_train = x_train.reshape(
            x_train.shape[0], img_w, img_h, img_channels)
        x_test = x_test.reshape(
            x_test.shape[0], img_w, img_h, img_channels)
        input_shape = (
            img_w, img_h, img_channels)

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test, input_shape)

def plot_training_history(history):
    """Draws a graph with the trainning history"""
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['validation accuracy'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
