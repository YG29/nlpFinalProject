import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Add

class Encoder(tf.keras.layers.Layer):
    """
    transformer encoder
    """
    def __int__(self, dimension, num_heads, hidden_dimension, dropout_rate = 0.2):
        super(Encoder, self).__init__()
        self.multiheadattention = MultiHeadAttention(num_heads, dimension)
        self.feedforward = tf.keras.Sequential([
            Dense(hidden_dimension, activation = 'relu')
            Dense(dimension)
        ])

        self.normlayer1 = LayerNormalization(epsilon = 1e-6)
        self.normlayer2 = LayerNormalization(epsilon = 1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_out, = self.multiheadattention(inputs, inputs, inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.normlayer1(x + attn_out)

        ff_out = self.feedforward(out1)
        ff_out = self.normlayer2(ff_out, training=training)
        out2 = self.normlayer2(out1 + ff_out)

        return out2