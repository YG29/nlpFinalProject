from src.transformer import Transformer
from src.utilfunc import create_mask

import numpy as np
import pandas as pd
import tensorflow as tf


TRAIN_PADDED = '../data/clean/train_padded.pkl'
VAL_PADDED = '../data/clean/val_padded.pkl'
checkpoint_path = '../data/checkpoint/'
BATCH_SIZE = 32
MAX_SENTENCE = 50

# parameters
num_layers = 2
embedding_dimension = 512
num_heads = 4
hidden_dimension = 4
input_vocab_size = 900
target_vocab_size = 900
learning_rate = 0.2
num_epochs = 200
clip_value_min = -1.0
clip_value_max = 1.0


def str_to_np_array(s):

    if isinstance(s, str):
        # Remove brackets and split by comma to get individual elements
        elements = s.strip('[]').split(', ')
        # Convert elements to float or int as required
        return np.array(elements, dtype=float)  # Change dtype as needed
    elif isinstance(s, np.ndarray):
        # If already a NumPy array, return it directly
        return s
    else:
        raise ValueError("Unsupported data type. Expected str or np.ndarray.")


train_dataset = pd.read_pickle(TRAIN_PADDED)
val_dataset = pd.read_pickle(VAL_PADDED)

train_dataset = train_dataset.map(str_to_np_array)
val_dataset = val_dataset.map(str_to_np_array)

train_inputs = train_dataset['inputs'].tolist()
train_targets = train_dataset['targets'].tolist()
val_inputs = val_dataset['inputs'].tolist()
val_targets = val_dataset['targets'].tolist()

# Create TensorFlow datasets from the lists
train_data = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
val_data = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))

# Padding and batching datasets
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([MAX_SENTENCE], [MAX_SENTENCE]), drop_remainder=True)
val_data = val_data.padded_batch(BATCH_SIZE, padded_shapes=([MAX_SENTENCE], [MAX_SENTENCE]), drop_remainder=True)

# model set up
model = Transformer(num_layers, embedding_dimension, num_heads, hidden_dimension, input_vocab_size, target_vocab_size, max_len_input=MAX_SENTENCE, max_len_output=MAX_SENTENCE, dropout_rate=0.1)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define checkpoint and early stopping
checkpoint_path = "./checkpoints/transformer.ckpt"
checkpoint = tf.train.Checkpoint(transformer=model)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# training loop
for epoch in range(num_epochs):
    total_train_loss = 0.0
    num_train_batches = 0

    for inputs, targets in train_data:
        print("Shape of input tensor:", inputs.shape)
        print("Shape of target tensor:", targets.shape)

        # create the masks
        encoding_padding_mask, future_mask, decoding_padding_mask = create_mask(inputs, targets)

        with tf.GradientTape() as tape:
            predictions, _, _ = model(inputs, targets, encoding_padding_mask, future_mask, decoding_padding_mask, training_bool=True)
            print("shape of predictions:", predictions.shape)
            loss = loss_object(targets, predictions)
            print("shape of loss:", loss.shape)
            total_train_loss += loss.numpy().sum()
            num_train_batches += 1

        gradients = tape.gradient(loss, model.trainable_variables)
        print("shape of gradient:", gradients.shape)
        # Gradient clipping
        clipped_gradients = [tf.clip_by_value(grad, clip_value_min, clip_value_max) for grad in gradients]

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # validation loop
    total_val_loss = 0.0
    num_val_batches = 0

    for inputs, targets in val_data:
        encoding_padding_mask, future_mask, decoding_padding_mask = create_mask(inputs, targets)
        predictions, _, _ = model(inputs, targets, encoding_padding_mask, future_mask, decoding_padding_mask, training_bool=False)
        val_loss = loss_object(targets, predictions)
        total_val_loss += val_loss.numpy().sum()
        num_val_batches += 1

    # print average train/validation loss
    ave_train_loss = total_train_loss / num_train_batches
    ave_val_loss = total_val_loss / num_val_batches
    print(f"Epoch {epoch + 1}, Train Loss: {ave_train_loss:.4f}, Validation Loss: {ave_val_loss:.4f}")

    # save checkpoint and early stopping
    checkpoint.save(file_prefix=checkpoint_path)
    early_stopping.on_epoch_end(ave_val_loss)
    if early_stopping.stopped_epoch == epoch:
        print(f"Early stopping at epoch {epoch + 1}")
        break
