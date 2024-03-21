import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Add

from src.positionalencoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder

class Transformer(tf.keras.Model):
    """
    set up the transformer architecture
    """

    def __init__(self, num_layers, dimension, num_heads, hidden_dimension,
                 input_vocab_size, target_vocab_size, max_len_input, max_len_output,
                 dropout_rate = 0.2):
        super(Transformer, self).__init__()
        self.encoder_embedding = Embedding(input_vocab_size, dimension)
        self.decoder_embedding = Embedding(target_vocab_size, dimension)
        self.pos_encoding = PositionalEncoding(max_len_input, dimension)
        self.decoder_pos_encoding = PositionalEncoding(max_len_output, dimension)

        self.encoder_layers = [Encoder(dimension, num_heads, hidden_dimension, dropout_rate)
                               for _ in range(num_layers)]
        self.decoder_layers = [Decoder(dimension, num_heads, hidden_dimension, dropout_rate)
                               for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, targets, training, mask):
        inpput_embedding = self.encoder_embedding(inputs)
        inpput_embedding *= tf.math.sqrt(tf.cast(self.dimension, tf.float32))
        input_embedding = self.PositionalEncode(input_embedding)

        target_embedding = self.decoder_embedding(targets)
        target_embedding *= tf.math.sqrt(tf.cast(self.dimension, tf.float32))
        target_embedding = self.decoder_pos_encoding(target_embedding)

        encoder_output = input_embedding
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, training, enc_padding_mask)

        dec_output = target_embedding
        for dec_layer in self.decoder_layers:
            dec_output, attn_weights1, attn_weights2 = dec_layer(dec_output, enc_output, training,
                                                                             look_ahead_mask, dec_padding_mask)

        dec_output = self.dropout(dec_output, training=training)
        final_output = self.final_layer(dec_output)

        return final_output, attn_weights1, attn_weights2