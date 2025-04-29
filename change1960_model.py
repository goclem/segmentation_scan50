#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Model structures for the Arthisto 1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
'''

# Modules
from keras import callbacks, layers, models, utils

# Convolution block
def convolution_block(input, filters:int, dropout:float, training:bool, name:str):
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, name=f'{name}_convolution1')(input)
    x = layers.Activation(activation='relu', name=f'{name}_activation1')(x)
    x = layers.BatchNormalization(name=f'{name}_normalisation1')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, name=f'{name}_convolution2')(x)
    x = layers.Activation(activation='relu', name=f'{name}_activation2')(x)
    x = layers.BatchNormalization(name=f'{name}_normalisation2')(x)
    x = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(x, training=training)
    return x

# Encoder block
def encoder_block(input, filters:int, dropout:float, training:bool, name:str):
    x = convolution_block(input=input, filters=filters, dropout=dropout, training=training, name=name)
    p = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(x)
    return x, p

# Decoder block
def decoder_block(input, skip, filters:int, dropout:float, training:bool, name:str):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name=f'{name}_transpose')(input)
    x = layers.Concatenate(name=f'{name}_concatenate')([x, skip])
    x = convolution_block(input=x, filters=filters, dropout=dropout, training=training, name=name)
    return x

# Decoder block
def binary_unet(input_shape:dict, filters:int, dropout:float, training:bool):
    # Input
    inputs = layers.Input(input_shape)
    # Encoder path
    s1, p1 = encoder_block(input=inputs, filters=1*filters, dropout=dropout, training=training, name='encoder1')
    s2, p2 = encoder_block(input=p1,     filters=2*filters, dropout=dropout, training=training, name='encoder2')
    s3, p3 = encoder_block(input=p2,     filters=4*filters, dropout=dropout, training=training, name='encoder3')
    s4, p4 = encoder_block(input=p3,     filters=8*filters, dropout=dropout, training=training, name='encoder4')
    # Bottleneck
    b1 = convolution_block(input=p4, filters=16*filters, dropout=dropout, training=training, name='bottleneck')
    # Decoder path
    d1 = decoder_block(input=b1, skip=s4, filters=8*filters, dropout=dropout, training=training, name='decoder1')
    d2 = decoder_block(input=d1, skip=s3, filters=4*filters, dropout=dropout, training=training, name='decoder2')
    d3 = decoder_block(input=d2, skip=s2, filters=2*filters, dropout=dropout, training=training, name='decoder3')
    d4 = decoder_block(input=d3, skip=s1, filters=1*filters, dropout=dropout, training=training, name='decoder4')
    # Output
    outputs = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(d4)
    # Model
    model = models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    return model

'''
# Displays structure
summary = DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in binary_unet.layers])
summary.style.to_html(path.join(paths['models'], 'unet64_structure.html'), index=False) 
del summary
tensorflow.keras.utils.plot_model(binary_unet, to_file=path.join(paths['models'], 'unet64_structure.pdf'), show_shapes=True)
'''
