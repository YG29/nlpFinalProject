import tensorflow as tf

def create_mask(inputs, targets):
    # add extra dimensions for 1-dimensional inputs
    inputs = inputs[:, tf.newaxis]
    targets = targets[:, tf.newaxis]

    # encoding mask
    encoding_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoding_padding_mask = encoding_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # decoding mask
    decoding_padding_mask = tf.cast(tf.math.equal(targets, 0), tf.float32)
    decoding_padding_mask = decoding_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # future mask
    future_mask = tf.cast(tf.linalg.band_part(tf.ones((tf.shape(targets)[1], tf.shape(targets)[1])), -1, 0), tf.float32)
    future_mask = tf.maximum(decoding_padding_mask, future_mask)

    return encoding_padding_mask, future_mask, decoding_padding_mask
