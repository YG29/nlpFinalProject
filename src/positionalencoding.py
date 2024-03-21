import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    """
    provide the positional encoding for each sentence
    """
    def __init__(self, max_length, dimension):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_length
        self.d_model = dimension
        self.pos_encode = self.positional_encoding(max_length, dimension)

    def get_angles(self, pos, idx, dimension):
        angle = 1 / tf.pow(10000, (2 * (idx // 2)) / tf.cast(dimension, tf.float32))
        return pos * angle

    def positional_encoding(self, max_length, dimension):
        angle_rad = self.get_angles(tf.range(max_length, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(dimension, dtype=tf.float32)[tf.newaxis, :],
                                     dimension)

        # sine is even
        angle_rad[:, 0::2] = tf.sin(angle_rad[:, 0::2])

        # cosine is odd
        angle_rad[:, 1::2] = tf.cos(angle_rad[:, 1::2])

        pos_encoding = tf.cast(angle_rad, dtype=tf.float32)
        return pos_encoding

    def call(self, inputs):
        return inputs + self.pos_encoding[:inputs.shape[1], :]


