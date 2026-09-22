from environment_and_utilities import *

try:
    import keras
    from keras.models import Sequential                                     # type: ignore
    from keras.layers import Dense, Dropout, Activation, Flatten            # type: ignore
    from keras.layers import Conv2D, MaxPooling2D, BatchNormalization       # type: ignore
    from keras import optimizers
    from keras import regularizers
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except (ImportError, AttributeError):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
except ImportError:
    from tensorflow import keras                                            # type: ignore
    from tensorflow.keras.models import Sequential                          # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten # type: ignore
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization # type: ignore
    from tensorflow.keras import optimizers                                 # type: ignore
    from tensorflow.keras import regularizers                              # type: ignore
    from tensorflow.keras.preprocessing.image import ImageDataGenerator     # type: ignore


class Network:
    def __init__(self):
        self.weight_decay = 0.0005
        self.x_shape = [100, 100, 3]  # Resized all examples to this size

        self.model = self.build_model()
        self.model.load_weights(initial_weights_path)

    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(32, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.4))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

    def predict(self, X):
        return self.model.predict(X)

    def fit2(self, X, y, X_val=None, y_val=None, train_epochs=100):
        # training parameters
        batch_size = 256
        maxepoches = min(100, train_epochs)
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 10

        def lr_scheduler(epoch):
            self.model.save_weights(save_weights_path)
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=maxepoches,
                       batch_size=batch_size, shuffle=True, callbacks=[reduce_lr], verbose=1)

    def fit(self, X, y, X_val=None, y_val=None, train_epochs=100):
        # training parameters
        batch_size = 64
        maxepoches = min(100, train_epochs)
        learning_rate = 0.001
        #lr_decay = 1e-6
        lr_drop = 10

        def lr_scheduler(epoch):
            self.model.save_weights(save_weights_path)
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X)

        historytemp = self.model.fit(datagen.flow(X, y, batch_size=batch_size),
                                               steps_per_epoch=X.shape[0] // batch_size,
                                               epochs=maxepoches,
                                               validation_data=(X_val, y_val), callbacks=[reduce_lr], verbose=1)

    def summary(self):
        self.model.summary()



def do_training(network, train_epochs=10):
    X_train = np.load(dataset_path+'X_train.npy')
    X_val = np.load(dataset_path+'X_val.npy')
    y_train = np.load(dataset_path+'y_train.npy')
    y_val = np.load(dataset_path+'y_val.npy')

    network.fit2(X_train, y_train, X_val, y_val, train_epochs)
    network.model.save_weights(save_weights_path)



#network = Network()
#network.summary()

#do_training(network, train_epochs=10)