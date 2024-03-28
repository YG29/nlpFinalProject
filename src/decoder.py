import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Add

from src.attention import CausalSelfAttention, CrossAttention
from src.encoder import FeedForward


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, embedding_dimension, num_heads, hidden_dimension, dropout_rate=0.1):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=embedding_dimension,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=embedding_dimension,
            dropout=dropout_rate)
        self.feedforward = FeedForward(embedding_dimension, hidden_dimension)

    def call(self, inputs, encoder_out):
        inputs = self.causal_self_attention(inputs=inputs)
        inputs = self.cross_attention(inputs=inputs, context=encoder_out)

        #Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        inputs = self.feedforward(inputs)  # (batch_size, max_length, embedding_dimension)`.
        return inputs



class Decoder(tf.keras.layers.Layer):
    """
    transformer decoder
    """

    def __init__(self, embedding_dimension, num_heads, hidden_dimension, dropout_rate=0.2):
        super(Decoder, self).__init__()

        self.multiheadattention1 = MultiHeadAttention(num_heads, embedding_dimension)
        self.multiheadattention2 = MultiHeadAttention(num_heads, embedding_dimension)

        self.feedforward = tf.keras.Sequential([
            Dense(hidden_dimension, activation='relu'),
            Dense(embedding_dimension)
        ])

        self.normlayer1 = LayerNormalization(epsilon=1e-6)
        self.normlayer2 = LayerNormalization(epsilon=1e-6)
        self.normlayer3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, input, encoder_out, future_mask, decoding_padding_mask, training_bool):
        attn_out1, attn_weights1 = self.multiheadattention1(input, input, input, future_mask)
        print("shape of decoder attn1:", attn_out1.shape)
        attn_out1 = self.dropout1(attn_out1, training=training_bool)
        print("shape of decoder dropout:", attn_out1.shape)
        out1 = self.normlayer1(attn_out1 + input)
        print("shape of decoder out1:", out1.shape)

        attn_out2, attn_weights2 = self.multiheadattention2(encoder_out, encoder_out, out1, decoding_padding_mask)
        attn_out2 = self.dropout2(attn_out2, training=training_bool)
        out2 = self.normlayer2(attn_out2 + out1)

        ff_out = self.feedforward(out2)
        ff_out = self.normlayer3(ff_out, training=training_bool)
        out3 = self.normlayer3(ff_out + out2)

        return out3, attn_weights1, attn_weights2

