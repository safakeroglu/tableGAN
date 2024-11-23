import math
import numpy as np
import tensorflow as tf

from utils import *

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name
        
    def __call__(self, x, train=True):
        return tf.keras.layers.BatchNormalization(
            epsilon=self.epsilon,
            momentum=self.momentum,
            name=self.name
        )(x, training=train)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])
    ], axis=3)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    return tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding='same',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev),
        bias_initializer='zeros',
        name=name
    )(input_)


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    filters = output_shape[-1]
    
    deconv_layer = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        bias_initializer='zeros',
        name=name
    )
    
    deconv = deconv_layer(input_)
    
    if with_w:
        return (
            deconv,
            deconv_layer.kernel,
            deconv_layer.bias
        )
    return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.keras.layers.LeakyReLU(alpha=leak, name=name)(x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    dense_layer = tf.keras.layers.Dense(
        units=output_size,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        bias_initializer=tf.keras.initializers.Constant(bias_start),
        name=scope
    )
    
    output = dense_layer(input_)
    
    if with_w:
        return output, dense_layer.kernel, dense_layer.bias
    return output


def masking(input_tensor, label_col, attrib_num):
    # Get input shape
    i_shape = tf.shape(input_tensor)
    batch_size = i_shape[0]
    
    # Flatten the input
    temp = tf.reshape(input_tensor, [batch_size, -1])
    t_shape = tf.shape(temp)
    
    # Create mask
    mask = np.zeros([batch_size, t_shape[1]])
    mask = np.equal(mask, mask)  # All True matrix
    
    # Mask label columns
    mask_col = label_col
    for i in range(t_shape[1] // attrib_num):
        mask[:, mask_col] = False
        mask_col += attrib_num
    
    # Convert mask to tensor and apply
    mask_tensor = tf.constant(mask)
    masked = tf.where(mask_tensor, temp, tf.zeros_like(temp))
    
    # Reshape back to original shape
    return tf.reshape(masked, i_shape)


class Logger:
    def __init__(self, log_dir):
        self.summary_writer = tf.summary.create_file_writer(log_dir)
    
    def log_scalar(self, name, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)
    
    def log_image(self, name, images, step):
        with self.summary_writer.as_default():
            tf.summary.image(name, images, step=step)
    
    def log_histogram(self, name, values, step):
        with self.summary_writer.as_default():
            tf.summary.histogram(name, values, step=step)
