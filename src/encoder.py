import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Add

from src.attention import GlobalSelfAttention
from src.positionalencoding import PositionalEncoding

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, hidden_dimension, dropout_rate=0.1):
        super().__init__()

        self.feedforward = tf.keras.Sequential([
            Dense(hidden_dimension, activation='relu'),
            Dense(embedding_dimension),
            Dropout(dropout_rate)
        ])
        self.add = Add()
        self.normlayer = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        inputs = self.add([inputs, self.feedforward(inputs)])
        inputs = self.normlayer(inputs)
        return inputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, embedding_dimension, num_heads, hidden_dimension, dropout_rate=0.1):
        super().__init__()

        self.multiheadattention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=embedding_dimension,
            dropout=dropout_rate)

        self.feedforward = FeedForward(embedding_dimension, hidden_dimension)

    def call(self, inputs):
        inputs = self.multiheadattention(inputs)
        inputs = self.feedforward(inputs)
        return inputs


class Encoder(tf.keras.layers.Layer):
    """
    transformer encoder
    """
    def __init__(self, num_layers, embedding_dimension, num_heads, hidden_dimension, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.pos_embedding = PositionalEncoding(
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension
        )
        self.encoding_layer = [
            EncoderLayer(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                hidden_dimension=hidden_dimension,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs): # input shape of (batch_size, max_length)

        inputs = self.pos_embedding(inputs) # shape(batch_size, max_length, embedding_dimension)
        inputs = self.dropout(inputs)

        for idx in range(self.num_layers):
            inputs = self.encoding_layer[idx](inputs)

        return inputs # out:(batch_size, max_length, embedding_dimension)

