from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras import backend as K

## Data Set: cifar10
## 0 - airplane 
## 1 - automobile 
## 2 - bird 
## 3 - cat 
## 4 - deer 
## 5 - dog 
## 6 - frog 
## 7 - horse 
## 8 - ship 
## 9 - truck

def main():
    """Main Function"""

    ## load data set
    x_train, y_train, x_test, y_test, input_shape = load_cifar10_dataset()

    ## create model
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    ## the sequential model is a linear stack of layers
    model = Sequential()

    ## Convolution operator for filtering windows of two-dimensional inputs
    model.add(Convolution2D(
        nb_filters, kernel_size[0], kernel_size[1], \
        border_mode='valid', input_shape=input_shape))

    ## Activation layer with relu
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    ## Configures the learning process
    model.compile(
        loss='categorical_crossentropy', \
        optimizer='adadelta', \
        metrics=['accuracy'])

    ## train the model
    model.fit(
        x_train, \
        y_train, \
        batch_size=64, \
        nb_epoch=3, \
        verbose=1, \
        validation_split=0.1)

    ## test the model
    score = model.evaluate(x_test, y_test)
    print "\nTest accuracy: %0.05f" % score[1]

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

if __name__ == '__main__':
    main()
