import tensorflow as tf
from tensorflow.keras.layers import Embedding
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding = Embedding(vocab_size, embedding_dimension, mask_zero=True)
        self.pos_encoding = self.positional_encoding(length=50, depth=embedding_dimension)

    def positional_encoding(length, depth):
        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        inputs = self.embedding(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        inputs *= tf.math.sqrt(tf.cast(self.embedding_dimension, tf.float32))
        inputs = inputs + self.pos_encoding[tf.newaxis, :length, :]
        return inputs


#     def get_angles(self, pos, idx, d_model):
#         angle_rates = 1 / tf.pow(10000, (2 * tf.cast(idx // 2, tf.float32)) / tf.cast(d_model, tf.float32))
#         return pos * angle_rates
#
#     def positional_encoding(self, max_len, embedding_dim):
#         depth = embedding_dim // 2
#
#         positions = np.arange(max_len)[:, np.newaxis]     # (seq, 1)
#         depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)
#
#         angle_rates = 1 / (10000 ** depths)         # (1, depth)
#         angle_rads = positions * angle_rates      # (pos, depth)
#
#         pos_encoding = np.concatenate(
#             [np.sin(angle_rads), np.cos(angle_rads)],
#             axis=-1)
#
#         return tf.cast(pos_encoding, dtype=tf.float32)
#
#     def call(self, inputs):
#         max_len = tf.shape(inputs)[1]  # Get the actual length of the sequence
#         return inputs + self.pos_encoding[:max_len, :]
#
#
# # class PositionalEncoding(tf.keras.layers.Layer):
# #     """
# #     provide the positional encoding for each sentence
# #     """
# #
# #     def __init__(self, max_length, embedding_dimension):
# #         super(PositionalEncoding, self).__init__()
# #         self.max_seq_len = max_length
# #         self.d_model = embedding_dimension
# #         self.pos_encoding = self.positional_encoding(max_length, embedding_dimension)
# #
# #     def get_angles(self, pos, idx, d_model):
# #         angle_rates = 1 / tf.pow(10000, (2 * tf.cast(idx // 2, tf.float32)) / tf.cast(d_model, tf.float32))
# #         # Casting pos to float32
# #         pos_float32 = tf.cast(pos, tf.float32)
# #         angle_rates_float32 = tf.cast(angle_rates, tf.float32)
# #
# #         return tf.cast(pos_float32 * angle_rates_float32, tf.int32)
# #
# #     def positional_encoding(self, max_length, embedding_dimension):
# #         position = tf.cast(tf.range(self.max_seq_len), tf.float32)
# #         indices = tf.cast(tf.range(self.d_model), tf.float32)
# #
# #         angle_rads = self.get_angles(position[:, tf.newaxis], indices[tf.newaxis, :], self.d_model)
# #         sin_indices = tf.range(0, self.d_model, 2)
# #         cos_indices = tf.range(1, self.d_model, 2)
# #
# #         angle_rads_float = tf.cast(angle_rads, tf.float32)  # Casting to float32
# #         angle_rads_float = tf.tensor_scatter_nd_update(angle_rads_float, tf.expand_dims(sin_indices, axis=-1),
# #                                                        tf.math.sin(angle_rads_float[:, 0::2]))
# #         angle_rads_float = tf.tensor_scatter_nd_update(angle_rads_float, tf.expand_dims(cos_indices, axis=-1),
# #                                                        tf.math.cos(angle_rads_float[:, 1::2]))
# #         pos_encoding = tf.expand_dims(angle_rads_float, axis=0)
# #
# #         return tf.cast(pos_encoding, tf.float32)
# #
# #     def call(self, inputs):
# #         batch_size = tf.shape(inputs)[0]
# #         return inputs + self.pos_encoding[:, :batch_size, :self.max_seq_len]
# #
# #
# #
# #
# #
# # # class PositionalEncoding(tf.keras.layers.Layer):
# # #     """
# # #     provide the positional encoding for each sentence
# # #     """
# # #     def __init__(self, max_length, embedding_dimension):
# # #         super(PositionalEncoding, self).__init__()
# # #         self.max_seq_len = max_length
# # #         self.d_model = embedding_dimension
# # #         self.pos_encoding = self.positional_encoding(max_length, embedding_dimension)
# # #
# # #     def get_angles(self, pos, idx, d_model):
# # #         angle_rates = 1 / tf.pow(10000, (2 * tf.cast(idx // 2, tf.float32)) / tf.cast(d_model, tf.float32))
# # #         return pos * angle_rates
# # #
# # #     def positional_encoding(self, max_length, embedding_dimension):
# # #         position = tf.cast(tf.range(self.max_seq_len), tf.int32)
# # #         indices = tf.cast(tf.range(self.d_model), tf.int32)
# # #
# # #         angle_rads = self.get_angles(position[:, tf.newaxis], indices[tf.newaxis, :], self.d_model)
# # #         sin_indices = tf.range(0, self.d_model, 2)
# # #         cos_indices = tf.range(1, self.d_model, 2)
# # #
# # #         angle_rads = tf.tensor_scatter_nd_update(angle_rads, tf.expand_dims(sin_indices, axis=-1),
# # #                                                  tf.math.sin(angle_rads[:, 0::2]))
# # #         angle_rads = tf.tensor_scatter_nd_update(angle_rads, tf.expand_dims(cos_indices, axis=-1),
# # #                                                  tf.math.cos(angle_rads[:, 1::2]))
# # #         pos_encoding = tf.expand_dims(angle_rads, axis=0)
# # #
# # #         return tf.cast(pos_encoding, tf.float32)
# # #
# # #         # angle_rads = self.get_angles(tf.range(self.max_seq_len)[:, tf.newaxis], tf.range(self.d_model)[tf.newaxis, :], self.d_model)
# # #         # sin_indices = tf.range(0, self.d_model, 2)
# # #         # cos_indices = tf.range(1, self.d_model, 2)
# # #         # angle_rads = tf.tensor_scatter_nd_update(angle_rads, tf.expand_dims(sin_indices, axis=-1),
# # #         #                                          tf.math.sin(angle_rads[:, 0::2]))
# # #         # angle_rads = tf.tensor_scatter_nd_update(angle_rads, tf.expand_dims(cos_indices, axis=-1),
# # #         #                                          tf.math.cos(angle_rads[:, 1::2]))
# # #         # pos_encoding = tf.expand_dims(angle_rads, axis=0)
# # #         # return tf.cast(pos_encoding, tf.float32)
# # #
# # #         # position = tf.cast(tf.range(self.max_seq_len), tf.float32)
# # #         # angle_rads = self.get_angles(position[:, tf.newaxis], tf.range(self.d_model)[tf.newaxis, :], self.d_model)
# # #         # # angle_rads = self.get_angles(tf.range(max_length, dtype=tf.float32)[:, tf.newaxis],
# # #         # #                              tf.range(embedding_dimension, dtype=tf.float32)[tf.newaxis, :],
# # #         # #                              embedding_dimension)
# # #         #
# # #         # # sine is even
# # #         # angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
# # #         #
# # #         # # cosine is odd
# # #         # angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
# # #         #
# # #         # pos_encoding = tf.reshape(angle_rads, [max_length, embedding_dimension])
# # #         #
# # #         # return tf.cast(pos_encoding, dtype=tf.float32)
# # #
# # #     def call(self, inputs):
# # #         batch_size = tf.shape(inputs)[0]
# # #         return inputs + self.pos_encoding[:batch_size, :self.max_seq_len]
# # #
