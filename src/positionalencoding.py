import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    """
    provide the positional encoding for each sentence
    """
    def __init__(self, max_length, embedding_dimension):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_length
        self.d_model = embedding_dimension
        self.pos_encoding = self.positional_encoding(max_length, embedding_dimension)

    def get_angles(self, pos, idx, embedding_dimension):
        angle = 1 / tf.pow(10000, (2 * (idx // 2)) / tf.cast(embedding_dimension, tf.float32))
        return pos * angle

    def positional_encoding(self, max_length, embedding_dimension):
        angle_rad = self.get_angles(tf.range(max_length, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(embedding_dimension, dtype=tf.float32)[tf.newaxis, :],
                                     embedding_dimension)

        # sine is even
        angle_sin = tf.math.sin(angle_rad[:, 0::2])

        # cosine is odd
        angle_cos = tf.math.cos(angle_rad[:, 1::2])

        pos_encoding = tf.stack([angle_sin, angle_cos], axis=2)
        pos_encoding = tf.reshape(pos_encoding, [max_length, embedding_dimension])

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return inputs + self.pos_encoding[:batch_size, :self.max_seq_len]

