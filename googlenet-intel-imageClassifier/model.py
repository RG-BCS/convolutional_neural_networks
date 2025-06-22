from tensorflow import keras

def build_inception_model(num_classes=6, image_size=(224, 224, 3)):
    layer_in = keras.layers.Input(shape=image_size)

    # Stage 1
    layer_ = keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(layer_in)
    layer_ = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(layer_)
    layer_ = keras.layers.BatchNormalization()(layer_)

    # Stage 2
    layer_ = keras.layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(layer_)
    layer_ = keras.layers.Conv2D(192, 1, strides=1, padding='same', activation='relu')(layer_)
    layer_ = keras.layers.BatchNormalization()(layer_)
    layer_ = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(layer_)

    # Inception modules here (same as your original code), with num_classes used below

    # Inception 1
    filters = [96,16,64,128,32,32]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Inception module 2
    filters = [128,32,128,192,96,64]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])
    layer_ = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(layer_)

    # Inception module 3
    filters = [96,16,192,208,48,64]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Auxiliary 1
    layer_aux1 = keras.layers.AveragePooling2D(pool_size=5, strides=3, padding='valid')(layer_)
    layer_aux1 = keras.layers.Conv2D(128, 1, padding='same', activation='relu')(layer_aux1)
    layer_aux1 = keras.layers.GlobalAveragePooling2D()(layer_aux1)
    layer_aux1 = keras.layers.Dense(256, activation='relu')(layer_aux1)
    layer_aux1 = keras.layers.Dropout(0.4)(layer_aux1)
    aux1_out = keras.layers.Dense(num_classes, activation='softmax')(layer_aux1)

    # Inception module 4
    filters = [112,24,160,224,64,64]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Inception module 5
    filters = [128,24,128,256,64,64]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Inception module 6
    filters = [144,32,112,288,64,64]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Auxiliary 2
    layer_aux2 = keras.layers.AveragePooling2D(pool_size=5, strides=3, padding='valid')(layer_)
    layer_aux2 = keras.layers.Conv2D(128, 1, padding='same', activation='relu')(layer_aux2)
    layer_aux2 = keras.layers.GlobalAveragePooling2D()(layer_aux2)
    layer_aux2 = keras.layers.Dense(256, activation='relu')(layer_aux2)
    layer_aux2 = keras.layers.Dropout(0.4)(layer_aux2)
    aux2_out = keras.layers.Dense(num_classes, activation='softmax')(layer_aux2)

    # Inception module 7
    filters = [160,32,256,320,128,128]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])
    layer_ = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(layer_)

    # Inception module 8
    filters = [160,32,256,320,128,128]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    # Inception module 9
    filters = [192,48,384,384,128,128]
    path_1 = keras.layers.Conv2D(filters[0], kernel_size=1, padding='same', activation='relu')(layer_)
    path_2 = keras.layers.Conv2D(filters[1], kernel_size=1, padding='same', activation='relu')(layer_)
    path_3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(layer_)

    path_4 = keras.layers.Conv2D(filters[2], kernel_size=1, padding='same', activation='relu')(layer_)
    path_5 = keras.layers.Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(path_1)
    path_6 = keras.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(path_2)
    path_7 = keras.layers.Conv2D(filters[5], kernel_size=1, padding='same', activation='relu')(path_3)
    layer_ = keras.layers.Concatenate()([path_4, path_5, path_6, path_7])

    layer_ = keras.layers.AveragePooling2D(pool_size=7, padding='valid')(layer_)
    layer_ = keras.layers.Flatten()(layer_)
    layer_ = keras.layers.Dropout(0.5)(layer_)
    layer_ = keras.layers.Dense(256, activation='linear')(layer_)
    layer_out = keras.layers.Dense(num_classes, activation='softmax', name='main')(layer_)

    model = keras.Model(inputs=[layer_in], outputs=[layer_out, aux1_out, aux2_out])
    return model
