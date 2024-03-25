import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization

from src.positionalencoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(tf.keras.Model):
    """
    set up the transformer architecture
    1. embedding
    2. add in the positional encoding
    3. encoder
    4. decoder
    """

    def __init__(self, num_layers, embedding_dimension, num_heads, hidden_dimension,
                 input_vocab_size, target_vocab_size, max_len_input, max_len_output,
                 dropout_rate=0.2):
        super(Transformer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.encoder_embedding = Embedding(input_vocab_size, embedding_dimension)
        self.decoder_embedding = Embedding(target_vocab_size, embedding_dimension)
        self.pos_encoding = PositionalEncoding(max_len_input, embedding_dimension)
        self.decoder_pos_encoding = PositionalEncoding(max_len_output, embedding_dimension)

        self.encoder_layers = [Encoder(embedding_dimension, num_heads, hidden_dimension, dropout_rate)
                               for _ in range(num_layers)]
        self.decoder_layers = [Decoder(embedding_dimension, num_heads, hidden_dimension, dropout_rate)
                               for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, input, target, encoding_padding_mask,
             future_mask, decoding_padding_mask, training_bool):
        input_embedding = self.encoder_embedding(input)
        input_embedding *= tf.math.sqrt(tf.cast(self.embedding_dimension, tf.float32))
        input_embedding = self.pos_encoding(input_embedding)

        target_embedding = self.decoder_embedding(target)
        target_embedding *= tf.math.sqrt(tf.cast(self.embedding_dimension, tf.float32))
        target_embedding = self.decoder_pos_encoding(target_embedding)

        encoder_output = input_embedding
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, encoding_padding_mask, training_bool)

        decoder_output = target_embedding
        for decoder_layer in self.decoder_layers:
            decoder_output, attn_weights1, attn_weights2 = decoder_layer(decoder_output, encoder_output, future_mask, decoding_padding_mask, training_bool)

        decoder_output = self.dropout(decoder_output, training=training_bool)
        final_output = self.final_layer(decoder_output)

        return final_output, attn_weights1, attn_weights2