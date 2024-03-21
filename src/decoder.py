import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Add

class Decoder(tf.keras.layers.Layer):
    """
    transformer decoder
    """

    def __int__(self):
        def __int__(self, dimension, num_heads, hidden_dimension, dropout_rate=0.2):
            super(Decoder, self).__init__()
            self.multiheadattention1 = MultiHeadAttention(num_heads, dimension)
            self.multiheadattention2 = MultiHeadAttention(num_heads, dimension)
            self.feedforward = tf.keras.Sequential([
                Dense(hidden_dimension, activation='relu')
                Dense(dimension)
            ])

            self.normlayer1 = LayerNormalization(epsilon=1e-6)
            self.normlayer2 = LayerNormalization(epsilon=1e-6)
            self.normlayer3 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
            self.dropout3 = Dropout(dropout_rate)

        def call(self, input, encoder_out, training, mask):
            attn_out1, attn_weights1 = self.multiheadattention1(input, input, input, mask)
            attn_out1 = self.dropout1(attn_out1, training=training)
            out1 = self.normlayer1(attn_out1+input)

            attn_out2, attn_weights2 = self.multiheadattention2(encoder_out, encoder_out, out1, mask)
            attn_out2 = self.dropout2(attn_out2, training=training)
            out2 = self.layernorm2(attn_out2 + out1)

            ff_out = self.feedforward(out2)
            ff_out = self.normlayer3(ff_out, training=training)
            out3 = self.normlayer3(out2 + ff_out)

            return out3, attn_weights1, attn_weights2

