import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


class CNN:
    """
    CNN class contains convolutional neural network models created using keras.
    """
    def __init__(self):
        pass

    def get_model(self, model_name):
        """ 
        Returns the model specified in model_name
        currently supports:
            fruits_classifier
        """
        layers = []
        if model_name == "fruit_classifier":
            layers.extend(self.fruit_classifier())
        elif model_name == "custom":
            layers.extend(self.custom())

        assert len(layers)!=0, "No layers in model."

        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def fruit_classifier(self, deep=0):
        """ Convolutional neural network classifer used in FruitCal project """
        layers = [
            # Convolutional Block 1
            Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # Convolutional Block 2
            Conv2D(192, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(192, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # Convolutional Block 3
            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            ### Commented out , but used for models 256_deeper_by_1 and 256_deeper_by_2

            # Conv2D(320, (3, 3), padding='same'),
            # BatchNormalization(),
            # Activation('relu'),
            # Conv2D(320, (3, 3), padding='same'),
            # BatchNormalization(),
            # Activation('relu'),
            # MaxPooling2D(pool_size=(2, 2)),
                
            # Conv2D(384, (3, 3), padding='same'),
            # BatchNormalization(),
            # Activation('relu'),
            # Conv2D(384, (3, 3), padding='same'),
            # BatchNormalization(),
            # Activation('relu'),
            # MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(256),                 
            BatchNormalization(),
            Activation('relu'),

            Dense(3),
            Activation('softmax')
        ]

        return layers

    def custom(self):
        layers = [
            Conv2D(128, (3,3), padding='same',input_shape=(128,128,3))
        ]

        return layers

