from src.transformer import Transformer
from src.utilfunc import create_mask

import numpy as np
import pandas as pd
import tensorflow as tf

TRAIN_PADDED = '../data/clean/train_padded.csv'
VAL_PADDED = '../data/clean/val_padded.csv'

BATCH_SIZE = 32
MAX_SENTENCE = 50

# parameters
num_layers = 2
embedding_dimension = 10
num_heads = 4
ff_dimension = 4
input_vocab_size = 900
target_vocab_size = 900
learning_rate = 0.2
num_epochs = 200


train_dataset = pd.read_csv(TRAIN_PADDED)
val_dataset = pd.read_csv(VAL_PADDED)
train_dataset['inputs'] = train_dataset['inputs'].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '))
train_dataset['targets'] = train_dataset['targets'].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '))
val_dataset['inputs'] = val_dataset['inputs'].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '))
val_dataset['targets'] = val_dataset['targets'].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '))

# def __init__(self, num_layers, embedding_dimension, num_heads, hidden_dimension,
#                  input_vocab_size, target_vocab_size, max_len_input, max_len_output,
#                  dropout_rate=0.2):

# model set up
model = Transformer(num_layers, embedding_dimension, num_heads, ff_dimension, input_vocab_size, target_vocab_size, max_len_input=MAX_SENTENCE, max_len_output=MAX_SENTENCE, dropout_rate=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate)

# training loop
for epoch in range(num_epochs):
    for index, row in train_dataset.iterrows():
        inputs = row['inputs']
        targets = row['targets']

        # create the masks
        encoding_padding_mask, future_mask, decoding_padding_mask = create_mask(inputs, targets)

        with tf.GradientTape() as tape:
            predictions, _, _ = model(inputs, targets, encoding_padding_mask, future_mask, decoding_padding_mask, training=True)
            loss = loss_object(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # validation loop
    total_val_loss = 0.0
    num_val_batches = 0
    for inputs, targets in val_dataset:
        encoding_padding_mask, future_mask, decoding_padding_mask = create_mask(inputs, targets)
        predictions, _, _ = model(inputs, targets, encoding_padding_mask, future_mask, decoding_padding_mask, training=False)
        val_loss = loss_object(targets, predictions)
        total_val_loss += val_loss.np().sum()
        num_val_batches += 1

    # print average validation loss
    ave_val_loss = total_val_loss / num_val_batches
    print(f"Epoch {epoch + 1}, Validation Loss: {ave_val_loss:.4f}")