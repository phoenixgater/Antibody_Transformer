# Transformer 1 - Primary structures only
# Feature: Antigen
# Label: Concatenated Heavy and light antibody variable domains
# Each amino acid residue is treated as a token in preprocessing
# A standard transformer architecture is used

# Importing modules
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
import time
import logo_maker
import Write_Results
import Query_SabDab
import Evaluation_And_Results.Antigen_Attention_Annotation as AAA

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# Antigen sequence for de novo inference
antigen_sequence_1 = "ESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP"


# Adds whitespace between each character of a string sequence
def add_whitespace(sequence):
    spaced_sequence = ""
    for aa in sequence:
        spaced_sequence += " " + aa
    spaced_sequence = spaced_sequence
    return spaced_sequence

# Function for creating a tensorflow dataset from a CSV file
def create_tf_dataset(csv_dataset_location):
    ds = pd.read_csv(csv_dataset_location, index_col=0)
    ds = ds.astype(str)
    ds_labels_raw = ds.copy()
    ds_features_raw = ds_labels_raw.pop("antigen")
    number_of_records = len(ds["concat_domains"])

    # Each tensor of labels will specify concatenated antibody variable domains
    ds_labels = pd.DataFrame(columns=["concat_domains"])
    for i in range(0, number_of_records):
        ds_labels.loc[i] = add_whitespace((ds_labels_raw["concat_domains"])[i])

    # Each tensor of features will specify an antigen
    ds_features = pd.DataFrame(columns=["antigen"])
    for i in range(0, number_of_records):
        ds_features.loc[i] = add_whitespace(ds_features_raw[i])

    print(ds_features)
    print(ds_labels)

    '''The tensorflow dataset is created by slicing across the dataframe
    specifying features, and the dataframe specifying labels, in parallel'''
    tf_dataset = tf.data.Dataset.from_tensor_slices((ds_features, ds_labels))
    print("Dataset created, size:", number_of_records)

    ''' We shuffle the dataset: Each feature tensor stays with its corresponding
     label tensor during this process'''
    tf_dataset = tf_dataset.shuffle(buffer_size=number_of_records,
                                    reshuffle_each_iteration=True)
    return [tf_dataset, number_of_records]


class Create_Vectorization_Layer:
    def __init__(self, vocab_size=22, inp_seq_len=500, out_seq_len=258):
        self.vocab_size = vocab_size
        self.inp_seq_len = inp_seq_len
        self.out_seq_len = out_seq_len

        self.vectorize_inp = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.inp_seq_len)

        self.vectorize_out = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.out_seq_len)

        vocabulary = ["l", "v", "g", "s", "t", "a", "i", "n", "e", "k", "d",
                      "p", "r", "f", "q", "y", "c", "h", "w", "m"]

        self.vectorize_inp.set_vocabulary(vocabulary)
        self.vectorize_out.set_vocabulary(vocabulary)

    # The text vectorization function is applied, and SOS and EOS are added
    def vectorize_tensor_pair(self, feature, label):
        vectorized_feature = np.insert(self.vectorize_inp(feature).numpy(), 0,
                                       self.vocab_size)
        vectorized_feature = np.insert(vectorized_feature,
                                       len(vectorized_feature),
                                       self.vocab_size + 1)

        # finding end of heavy variable domain
        label_2 = str(label.numpy())
        label_2 = label_2.replace("[", "").replace("]", "").replace("b", "").replace("'", "")
        label_2_split = label_2.split()
        j_i = "Error"
        for i, token in enumerate(label_2_split):
            if token == "J":
                j_i = i
                break
        j_i += 1  # first index after heavy variable domain, after SOS token has been added

        label_2 = label_2.replace("J", "")
        label_3 = label_2.replace(" ", "")
        eod_i = len(label_3) + 2  # first index after light variable domain - offset by two for SOS and SEP
        new_label = ""
        for token in label_2:
            if token == " ":
                pass
            else:
                new_label += token
                new_label += " "
        new_label = new_label.strip()

        new_label = tf.convert_to_tensor(new_label)
        new_label = tf.expand_dims(new_label, 0)

        vectorized_label = np.insert(self.vectorize_out(new_label).numpy(),
                                     0, self.vocab_size)
        vectorized_label = np.insert(vectorized_label, len(vectorized_label),
                                     self.vocab_size + 1)

        # X replaced with padding
        vectorized_label = np.where(vectorized_label == 1, 0, vectorized_label)

        # inserting special <SEP> token
        vectorized_label = np.insert(vectorized_label, j_i, self.vocab_size + 2)

        # inserting special <EOD> token
        vectorized_label = np.insert(vectorized_label, eod_i, self.vocab_size + 3)

        return vectorized_feature, vectorized_label

    # The vectorization process is wrapped into a tensorflow operation
    def wrap_vectorize_tensor_pair(self, feature, label):
        vectorized_feature, vectorized_label = tf.py_function(self.vectorize_tensor_pair,
                                                              [feature, label],
                                                              [tf.int64, tf.int64])
        vectorized_feature.set_shape([self.inp_seq_len + 2])
        vectorized_label.set_shape([self.out_seq_len + 4])

        return vectorized_feature, vectorized_label

    # We map the process to every feature and label pair in the dataset
    def vectorize_dataset(self, dataset):
        return dataset.map(self.wrap_vectorize_tensor_pair)


def batch_and_optimise(dataset, size, batch_size=10):
    dataset.shuffle(size)  # Samples are shuffled
    dataset = dataset.cache()  # Speeds up reloading of samples after first epoch
    dataset = dataset.batch(batch_size, drop_remainder=True)  # Creating mini-batches
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE)  # Next mini-batch prepared while current is still being processed by the model
    return dataset


# calculating angles for positional encoding
def get_angles(max_pos, i, embedding_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size))
    return max_pos * angle_rates


# Creating tensor of sinusoidal positional encodings
def positional_encoding(max_pos, embedding_size):
    angle_rads = get_angles(np.arange(max_pos)[:, np.newaxis],
                            np.arange(embedding_size)[np.newaxis, :],
                            embedding_size)

    # applying sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # applying cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Creating padding mask for a batch of sequences
def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    # Matrix multiplication of queries and keys
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # Scaling qk by dk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Applying a mask to the scaled attention logits
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Computing the value weights by applying softmax function
    value_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # Computing output by multiplying values by their value weights
    attention_output = tf.matmul(value_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

    return attention_output, value_weights


# Multihead attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # User-specified quantity of attention heads
        self.embedding_size = embedding_size  # The user-specified embedding size

        assert embedding_size % self.num_heads == 0

        self.depth = embedding_size // self.num_heads

        # Linear layers for projection of queries, keys, and values
        self.wq = tf.keras.layers.Dense(embedding_size)
        self.wk = tf.keras.layers.Dense(embedding_size)
        self.wv = tf.keras.layers.Dense(embedding_size)

        # Final linear layer for projection of concatenated output of all heads
        self.dense = tf.keras.layers.Dense(embedding_size)

    ''' The tensors of linearly projected queries, keys, and values, are reshaped
    to facilitate distinct attention computations with respect to each attention head
    (dimension "num_heads")'''

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # linear projection of queries keys and values
        q = self.wq(q)  # (batch_size, seq_len, embedding_size)
        k = self.wk(k)  # (batch_size, seq_len, embedding_size)
        v = self.wv(v)  # (batch_size, seq_len, embedding_size)

        # Reshaping tensors so that attention can be computed for respective heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # attention_output.shape: (batch_size, num_heads, seq_len_q, depth_v)
        # value_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_output, value_weights = scaled_dot_product_attention(
            q, k, v, mask)

        attention_output = tf.transpose(attention_output,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(attention_output,
                                      (batch_size, -1, self.embedding_size))  # (batch_size, seq_len_q, embedding_size)

        # linear projection of concatenated attention outputs for all heads
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, embedding_size)

        return output, value_weights


# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, ffn_width, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(embedding_size, num_heads)
        self.dense_1 = tf.keras.layers.Dense(ffn_width, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(embedding_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, enc_padding_mask):
        # Self attention sublayer
        attn_output, enc_block_1 = self.mha(x, x, x, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_size)

        # Node dropout for self attention sublayer
        attn_output = self.dropout1(attn_output, training=training)

        # Layer normalisation and addition of residual connection identity
        out1 = self.layernorm1(x + attn_output)  # (batch_size, inp_seq_len, embedding_size)

        # Dense sublayers
        dense_1_output = self.dense_1(out1)  # (batch_size, inp_seq_len, ffn_width)
        dense_2_output = self.dense_2(dense_1_output)  # (batch_size, inp_seq_len, embedding_size)

        # Node dropout for dense sublayer 2
        dense_2_output = self.dropout2(dense_2_output, training=training)

        # Layer normalisation and addition of residual connection identity
        dense_2_output = self.layernorm2(out1 + dense_2_output)  # (batch_size, inp_seq_len, embedding_size)

        return dense_2_output, enc_block_1


# Decoder layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, ffn_width, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(embedding_size, num_heads)
        self.mha2 = MultiHeadAttention(embedding_size, num_heads)

        self.dense_1 = tf.keras.layers.Dense(ffn_width, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(embedding_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             combined_mask, dec_padding_mask):
        # Self-attention sublayer (queries, keys, and values, from x)
        attn1, value_weights_block1 = self.mha1(x, x, x, combined_mask)  # (batch_size, out_seq_len, embedding_size)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Global-attention sublayer (values and keys from enc_output, queries from previous self-attention sublayer)
        attn2, value_weights_block2 = self.mha2(
            enc_output, enc_output, out1, dec_padding_mask)  # (batch_size, out_seq_len, embedding_size)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, out_seq_len, embedding_size)

        # Dense sublayers
        dense_1_output = self.dense_1(out2)  # (batch_size, out_seq_len, embedding_size)
        dense_2_output = self.dense_2(dense_1_output)
        dense_2_output = self.dropout3(dense_2_output, training=training)
        out3 = self.layernorm3(dense_2_output + out2)  # (batch_size, out_seq_len, embedding_size)

        '''returns the output tensor of representations for the decoder layer, 
           and the value weights that were computed for the self-attention 
           (block1) and global attention (block2) sublayers'''
        return out3, value_weights_block1, value_weights_block2


# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_encoder_layers, embedding_size, num_heads, ffn_width, vocab_size,
                 max_pos_inp, dropout_rate):
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.num_encoder_layers = num_encoder_layers  # quantity of encoder layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.pos_encoding = positional_encoding(max_pos_inp, self.embedding_size)

        # Instantiating all encoder layers, and specifying them by a list
        self.enc_layers = [EncoderLayer(embedding_size, num_heads, ffn_width, dropout_rate)
                           for _ in range(num_encoder_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, enc_padding_mask):
        inp_seq_len = tf.shape(x)[1]  # Input sequence length
        value_weights_enc = {}

        # Embedding input sequence of token representations
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_size)

        # Scaling the embedded token representations
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))

        # Element-wise addition of positional encodings
        x += self.pos_encoding[:, :inp_seq_len, :]

        x = self.dropout(x, training=training)

        ''' The first tensor of representations is passed to the first encoder
        layer, and the output of that layer is passed to the next encoder layer, 
        etc...'''
        for i in range(self.num_encoder_layers):
            x, enc_block_1 = self.enc_layers[i](x, training, enc_padding_mask)
            value_weights_enc['encoder_layer{}_block1'.format(i + 1)] = enc_block_1

        # The final output of the encoder
        return x, value_weights_enc  # (batch_size, input_seq_len, embedding_size)


# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_decoder_layers, embedding_size, num_heads, ffn_width, vocab_size,
                 max_pos_out, dropout_rate):
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.decoder_layers = num_decoder_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.pos_encoding = positional_encoding(max_pos_out, embedding_size)

        # instantiating all decoder layers, and specifying them as a list
        self.dec_layers = [DecoderLayer(embedding_size, num_heads, ffn_width, dropout_rate)
                           for _ in range(num_decoder_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             combined_mask, dec_padding_mask):
        out_seq_len = tf.shape(x)[1]

        # A dictionary in which value weights are to be stored
        value_weights = {}

        # embedding and positional encoding of (target) output sequence
        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_size)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.pos_encoding[:, :out_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.decoder_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   combined_mask, dec_padding_mask)

            value_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            value_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, embedding_size)
        return x, value_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_encoder_layers, num_decoder_layers, embedding_size, num_heads, ffn_width, vocab_size,
                 max_pos_inp, max_pos_out, dropout_rate):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, embedding_size, num_heads, ffn_width,
                               vocab_size, max_pos_inp, dropout_rate)

        self.decoder = Decoder(num_decoder_layers, embedding_size, num_heads, ffn_width,
                               vocab_size, max_pos_out, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, tar_out, training, enc_padding_mask,
             combined_mask, dec_padding_mask):
        enc_output, value_weights_enc = self.encoder(inp, training,
                                                     enc_padding_mask)  # (batch_size, inp_seq_len, embedding_size)

        # dec_output.shape == (batch_size, out_seq_len, embedding_size)
        dec_output, value_weights = self.decoder(
            tar_out, enc_output, training, combined_mask, dec_padding_mask)

        pred_out = self.final_layer(dec_output)  # (batch_size, out_seq_len, embedding_size)

        return pred_out, value_weights, value_weights_enc


# Custom learning rate schedule for Adam optimiser
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_size = embedding_size
        self.embedding_size = tf.cast(self.embedding_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)


# Computation of masked loss
def loss_function(real_out, pred_out):
    # tar_out.shape = (batch_size, out_seq_len -1)
    # pred_out.shape = (batch_size, out_seq_len -1, vocab_size)

    ''' Padding mask created with respect to the true/target output
    sequence'''
    mask = tf.math.logical_not(tf.math.equal(real_out, 0))  # batch_size, out_seq_len -1

    # Loss coefficient tensor
    '''loss_multiplier = tf.math.equal(real_out, vectorization_layer.vocab_size + 2)
    tf.print(loss_multiplier)
    one_tensor = tf.ones(tf.shape(real_out))
    loss_multiplier += one_tensor'''

    # Computing tensor containing loss incurred by each predicted label
    loss = loss_object(real_out, pred_out)  # batch_size, out_seq_len -1

    # Masking of padded position in loss tensor using padding mask
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask  # batch_size, out_seq_len -1

    '''The elements of the loss tensor are summed, as are the elements
    of the padding mask. The sum of the loss tensor is scaled with respect
    to the quantity of positions that were not padded, using the sum of the 
    elements of the padding mask'''
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# Computation of masked accuracy
def accuracy_function(real_out, pred_out):
    '''Computing a tensor where positions for which tar_out and pred_out match
       are specified by element "1", and are otherwise "0"'''
    accuracies = tf.equal(real_out, tf.argmax(pred_out, axis=2))

    # Creating mask with respect to padded positions in target output sequences
    mask = tf.math.logical_not(tf.math.equal(real_out, 0))

    '''Applying padding mask with "tf.math.logical_and": if True (1) and True (1)
     then True (1), otherwise False (0)'''
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # elements of accuracy tensor are summed, and this is scaled by sum of mask
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# "inp" is the mini-batch of input sequences and "tar" is the mini-batch of target output sequences
def create_masks(inp, tar):
    '''Masks padded positions with respect to the input sequences, in the encoder
    multi-head self attention sublayers'''
    enc_padding_mask = create_padding_mask(inp)

    ''' Masks padded positions with respect to the input sequences, in the decoder
    multi-head global attention sublayers'''
    dec_padding_mask = create_padding_mask(inp)

    '''Masks future positions in the target output sequence with respect to
    each position in the target output sequence, in the decoder self attention 
    sublayers'''
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

    '''Masks padded positions with respect to the target output sequence, in the
    decoder self attention sublayers'''
    dec_target_padding_mask = create_padding_mask(tar)

    # Combined padding and lookahead mask for decoder self-attention sublayers
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# Creating training signature
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # input sequence tensor
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # output sequence tensor
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_out = tar[:, :-1]  # mini-batch of target output sequences with EOS removed
    real_out = tar[:, 1:]  # mini-batch of target output sequences with SOS removed

    # Creation of masks for encoder and decoder
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_out)

    # tf.GradientTape() tracks the gradients of each trainable parameter
    with tf.GradientTape() as tape:
        predictions, _, _ = transformer(inp, tar_out,
                                        True,  # (training = True) dropout is used
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask)
        loss = loss_function(real_out, predictions)

    # The gradient of the loss with respect to each trainable parameter is computed
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # The trainable parameters are adjusted by the optimization algorithm
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # These are used to store accuracy and loss metrics that we can later inspect
    train_loss(loss)
    train_accuracy(accuracy_function(real_out, predictions))


# Validation and testing
def validation_and_testing(inp, tar, type):
    tar_out = tar[:, :-1]
    real_out = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_out)

    predictions, _, _ = transformer(inp, tar_out,
                                    False,  # (training = False): no dropout used
                                    enc_padding_mask,
                                    combined_mask,
                                    dec_padding_mask)
    loss = loss_function(real_out, predictions)

    '''Storing loss and accuracy metrics for our validation and test datasets 
    for later inspection'''
    if type == "val":
        val_loss(loss)
        val_accuracy(accuracy_function(real_out, predictions))
    elif type == "test":
        test_loss(loss)
        test_accuracy(accuracy_function(real_out, predictions))


def predict_out_seq(inp_seq, epoch):
    '''The input to the encoder (input sequence) is the preprocessed antigen
  (SARS-CoV-2 RBD) primary structure'''
    encoder_input = inp_seq

    # The first input to the decoder is the start of sequence (SOS) token
    decoder_input = [vectorization_layer.vocab_size]

    '''This is the variable that is going to be passed through for the
  autoregression: it will specify what has been predicted of the output sequence
  so far'''
    output = tf.expand_dims(decoder_input, 0)

    '''A dictionary we are going to use to store probabilities for predicted
  amino acid residues (tokens), for later inspection'''
    aa_probabilities = {}

    # Loop for autogressive prediction of output sequence
    for i in range(vectorization_layer.out_seq_len + 2):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        ''' "cur_seq_len" is the current length of the output sequence that has 
      been predicted so far. e.g. first iteration of loop this will be 2, the 
      next it will be 3, etc'''
        # predictions.shape == (1, cur_seq_len, vocab_size)
        predictions, value_weights, value_weights_enc = transformer(encoder_input,
                                                                    output,
                                                                    False,
                                                                    enc_padding_mask,
                                                                    combined_mask,
                                                                    dec_padding_mask)

        ''' Retrieve token probabilities for the last position of what has been 
      predicted of the output sequence so far. '''
        last_prediction = predictions[:, -1:, :]  # (1, 1, vocab_size)
        probablities = tf.nn.softmax(last_prediction, axis=-1)  # (1, 1, vocab_size)
        probablities = probablities.numpy()
        all_probabilities[epoch][i] = probablities

        '''Storing the highest token probability for the last predicted position
      for later inspection'''
        max_prob = np.amax(probablities)
        aa_probabilities[i] = [max_prob]

        '''Retrieving the vocabulary index of the highest probability token for 
      the last predicted position of the output sequence'''
        predicted_token_id = tf.cast(tf.argmax(last_prediction, axis=-1),
                                     tf.int32)  # (1)

        # return the output sequence if the last predicted token is an EOS token
        if predicted_token_id == vectorization_layer.vocab_size + 1:
            return tf.squeeze(output, axis=0), value_weights, aa_probabilities, value_weights_enc

        '''concatentating the vocab index of the highest probability token to the 
      output, which is given to the decoder as its input in the next iteration'''
        output = tf.concat([output, predicted_token_id], axis=-1)

    ''' return the output sequence if the max_out_len is reached, and the EOS 
      token still has not been predicted'''
    return tf.squeeze(output, axis=0), value_weights, aa_probabilities, value_weights_enc


def predict_fv(antigen_seq, epoch, result_id=1):
    tokenised_seq = add_whitespace(list(antigen_seq.lower()))
    tokenised_seq = tf.convert_to_tensor(tokenised_seq)
    tokenised_seq = tf.expand_dims(tokenised_seq, 0)
    preproc_seq = vectorization_layer.vectorize_inp(tokenised_seq)
    preproc_seq = np.insert(preproc_seq.numpy(), 0, vectorization_layer.vocab_size)
    preproc_seq = np.insert(preproc_seq, len(preproc_seq), vectorization_layer.vocab_size + 1)
    preproc_seq = tf.expand_dims(preproc_seq, 0)

    result, value_weights, aa_probabilities, value_weights_enc = predict_out_seq(preproc_seq, epoch)
    vocabulary = vectorization_layer.vectorize_inp.get_vocabulary()

    result = result.numpy()
    predicted_fv_design = ""
    seq_index = 0
    sep_token_i = 1
    eod_token_i = 1
    for i in result:
        if i < vectorization_layer.vocab_size:
            aa_probabilities[seq_index].append(vocabulary[i])
            predicted_fv_design += vocabulary[i]
            seq_index += 1
        elif i == vectorization_layer.vocab_size + 2:
            sep_token_i = seq_index
            seq_index += 1
        elif i == vectorization_layer.vocab_size + 3:
            eod_token_i = seq_index
            seq_index += 1
            break
        else:
            pass

    plot_all_heads(antigen_seq, predicted_fv_design, value_weights, epoch, sep_token_i, eod_token_i, value_weights_enc)
    # Attention annotations, functions from script (Antigen_Attention_Annotation.py)

    if att_dist_eval:
        AAA.create_attention_dist_logo(value_weights, antigen_seq, num_decoder_layers, num_heads, vocabulary, epoch,
                                       write_directory, "STD")
        AAA.create_node_edge_att(value_weights, antigen_seq, predicted_fv_design, num_decoder_layers, num_heads, epoch,
                                 write_directory)

    print(predicted_fv_design.upper())
    print('Antigen: {}'.format(antigen_seq))
    if sep_token_i != "unset":
        if eod_token_i != "unset":
            print('Predicted Fv heavy domain design: {}'.format(predicted_fv_design.upper()[:sep_token_i]))
            print('Predicted Fv light domain design: {}'.format(
                predicted_fv_design.upper()[sep_token_i: eod_token_i]))
            heavy_domain_seq = predicted_fv_design.upper()[:sep_token_i]
            light_domain_seq = predicted_fv_design.upper()[sep_token_i: eod_token_i]
            if result_id == 1:
                written_results["predicted_fv_heavy_1"].append(heavy_domain_seq)
                written_results["predicted_fv_light_1"].append(light_domain_seq)
            else:
                written_results["predicted_fv_heavy_2"].append(heavy_domain_seq)
                written_results["predicted_fv_light_2"].append(light_domain_seq)

            matched_fv_results, matched_cdr_results, matched_cdrh3_results = Query_SabDab.query_sabdab(
                heavy_domain_seq.replace("X", ""),
                light_domain_seq.replace("X", ""))

            if result_id == 1:
                written_results["sabdab_query_whole_1"].append(matched_fv_results)
                written_results["sabdab_query_cdr_1"].append(matched_cdr_results)
                written_results["sabdab_query_cdrh3_1"].append(matched_cdrh3_results)
            else:
                written_results["sabdab_query_whole_2"].append(matched_fv_results)
                written_results["sabdab_query_cdr_2"].append(matched_cdr_results)
                written_results["sabdab_query_cdrh3_2"].append(matched_cdrh3_results)

            logo_maker.create_probability_logo(aa_probabilities, vocabulary, write_directory, epoch,
                                               result_id=result_id, sep_id=sep_token_i, eod_id=eod_token_i)
        else:
            print("eod token not found")
    else:
        print("sep token not found")


# Plot training and validation loss
def plot_loss(train_history, val_history):
    train_loss_list = []
    val_loss_list = []
    for key in train_history.keys():
        train_loss_list.append(train_history[key][0])
    for key in val_history.keys():
        val_loss_list.append(val_history[key][0])

    plt.figure()
    plt.plot(train_loss_list, label='loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.ylim([0, 5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    Write_Results.write_image(plt, "loss_graph", write_directory, max(EPOCHS))


# Plot training and validation accuracy
def plot_accuracy(train_history, val_history):
    train_acc_list = []
    val_acc_list = []
    for key in train_history.keys():
        train_acc_list.append(train_history[key][1])
    for key in val_history.keys():
        val_acc_list.append(val_history[key][1])

    plt.figure()
    plt.plot(train_acc_list, label='accuracy')
    plt.plot(val_acc_list, label='val_accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    Write_Results.write_image(plt, "acc_graph", write_directory, max(EPOCHS))


# Plot heat map for a single attention head
def plot_attention_head(antigen_sequence, predicted_fv_sequence, head, head_identity, epoch, layer_number, sep_id,
                        eod_id):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    head_weights_no_pad = tf.slice(head, [0, 0], [eod_id +1, len(antigen_sequence) + 2])
    ax.matshow(head_weights_no_pad)

    inp_seq_proc = antigen_sequence

    inp_seq_proc = ["<SOS>"] + [aa.upper() for aa in inp_seq_proc] + ["<EOS>"]
    predicted_fv_sequence = ([aa.upper() for aa in predicted_fv_sequence[:sep_id]] + ["<SEP>"] +
                             [aa.upper() for aa in predicted_fv_sequence[sep_id: eod_id]] + ["<EOD>"])

    ax.set_xticks(range(len(antigen_sequence) + 2))
    ax.set_yticks(range(len(predicted_fv_sequence)))

    x_labels = []
    for pos, aa in enumerate(inp_seq_proc):
        if aa == "<SOS>" or aa == "<EOS>":
            x_labels.append(aa)
            continue
        elif pos == len(antigen_sequence) + 2:
            x_labels.append(aa)
            break
        elif pos % 10 == 0:
            x_labels.append(aa + " #" + str(pos))
        else:
            x_labels.append(aa)

    y_labels = []
    for pos, aa in enumerate(predicted_fv_sequence):
        pos += 1
        if aa == "<SOS>" or aa == "<EOS>":
            y_labels.append(aa)
            continue
        elif pos % 10 == 0:
            y_labels.append(("#" + str(pos)) + " " + aa)
        else:
            y_labels.append(aa)

    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Head {}'.format(str(head_identity)))

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False, )

    Write_Results.write_image(plt, "attention_head", write_directory, epoch, layer_number)
    plt.close(fig)


# Plots heatmap for self-attention head
def plot_self_att_head(sequence, head, head_identity, epoch, layer_number, sep_id, eod_id, type):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    if type == "encoder":
        head_weights_no_pad = tf.slice(head, [0, 0], [len(sequence) + 2, len(sequence) + 2])
        seq_proc = ["<SOS>"] + [aa.upper() for aa in sequence] + ["<EOS>"]
        ticks = len(sequence) + 2
    else:
        head_weights_no_pad = tf.slice(head, [0, 0], [eod_id + 1, eod_id + 2])
        seq_proc = ([aa.upper() for aa in sequence[:sep_id]] + ["<SEP>"] +
                    [aa.upper() for aa in sequence[sep_id: eod_id]] + ["<EOD>"])

        ticks = len(seq_proc)

    ax.matshow(head_weights_no_pad)

    ax.set_yticks(range(ticks))
    if type == "decoder":
        ax.set_xticks(range(ticks + 1))
    else:
        ax.set_xticks(range(ticks))

    if type == "decoder":
        seq_proc = ["<SOS>"] + seq_proc

    x_labels = []
    for pos, aa in enumerate(seq_proc):
        if aa == "<SOS>" or aa == "<EOS>":
            x_labels.append(aa)
            continue
        elif pos == ticks:
            x_labels.append(aa)
            break
        elif pos % 10 == 0:
            x_labels.append(aa + " #" + str(pos))
        else:
            x_labels.append(aa)

    if type == "decoder":
        seq_proc = seq_proc[1:]

    y_labels = []
    for pos, aa in enumerate(seq_proc):
        pos += 1
        if aa == "<SOS>" or aa == "<EOS>":
            y_labels.append(aa)
            continue
        elif pos % 10 == 0:
            y_labels.append(("#" + str(pos)) + " " + aa)
        else:
            y_labels.append(aa)

    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Head {}'.format(str(head_identity)))

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False, )

    if type == "encoder":
        image_id = "self_att_head_enc"
    else:
        image_id = "self_att_head_dec"

    Write_Results.write_image(plt, image_id, write_directory, epoch, layer_number)
    plt.close(fig)


# Plot heat map for all attention heads in a given multi-head attention layer
def plot_all_heads(antigen_sequence, predicted_fv_sequence, attention_dict, epoch, sep_id, eod_id, att_dict_enc):
    if all_heatmaps:
        for layer_number in range(1, num_decoder_layers + 1):
            attention_heads = tf.squeeze(attention_dict['decoder_layer' + str(layer_number) + '_block2'], 0)
            for h, head in enumerate(attention_heads):
                head_identity = h + 1
                plot_attention_head(antigen_sequence, predicted_fv_sequence, head, head_identity, epoch,
                                    str(layer_number), sep_id, eod_id)

        for layer_number in range(1, num_decoder_layers + 1):
            attention_heads = tf.squeeze(attention_dict['decoder_layer' + str(layer_number) + '_block1'], 0)
            for h, head in enumerate(attention_heads):
                head_identity = h + 1
                plot_self_att_head(predicted_fv_sequence, head, head_identity, epoch, str(layer_number), sep_id, eod_id,
                                   'decoder')

        for layer_number in range(1, num_decoder_layers + 1):
            attention_heads = tf.squeeze(att_dict_enc['encoder_layer' + str(layer_number) + '_block1'], 0)
            for h, head in enumerate(attention_heads):
                head_identity = h + 1
                plot_self_att_head(antigen_sequence, head, head_identity, epoch, str(layer_number), 0, 0, 'encoder')

    else:
        attention_heads = tf.squeeze(
            attention_dict['decoder_layer' + str(num_decoder_layers) + '_block2'], 0)

        for h, head in enumerate(attention_heads):
            head_identity = h + 1
            plot_attention_head(antigen_sequence, predicted_fv_sequence, head, head_identity, epoch,
                                str(num_decoder_layers), sep_id, eod_id)


# Single training run configuration
if '__main__' == __name__:
    dataset_identity = "ds_19"
    num_encoder_layers = 3
    num_decoder_layers = 3
    embedding_size = 240
    ffn_width = 250
    num_heads = 5
    dropout_rate = 0.3
    EPOCHS = [5]
    all_heatmaps = True
    att_dist_eval = False

    schedule_identity = "0"
    write_directory = Write_Results.assign_local_directory(schedule_identity)

# Training run as part of a schedule (run from Training_Schedule.py)
if '__main__' != __name__:
    from Training_Schedule import schedule_root, schedule_selected

    schedule_root = schedule_root
    schedule_selected = schedule_selected
    schedule_identity = schedule_selected.replace("schedule_", "")[:-4]
    write_directory = Write_Results.assign_local_directory(schedule_identity)

    schedule = pd.read_csv(schedule_root + "\\" + schedule_selected)
    for i, training_run in schedule.iterrows():
        if training_run["status"] == "current":
            schedule = schedule_selected.split("_")[1][:-4]
            dataset_identity = training_run["dataset_identity"]
            num_encoder_layers = training_run["encoder_layers"]
            num_decoder_layers = training_run["decoder_layers"]
            embedding_size = training_run["embedding_dim"]
            ffn_width = training_run["ffn_width"]
            num_heads = training_run["num_heads"]
            dropout_rate = training_run["dropout_rate"]
            all_heatmaps = True
            att_dist_eval = False
            EPOCHS_raw = training_run["EPOCHS"]
            EPOCHS = []
            for epoch in EPOCHS_raw.split(", "):
                EPOCHS.append(int(epoch))
            break

# Creating a dictionary, for written results
written_results = {"predicted_fv_heavy_1": [], "predicted_fv_light_1": [], "sabdab_query_whole_1": [],
                   "sabdab_query_cdr_1": [], "sabdab_query_cdrh3_1": [], "predicted_fv_heavy_2": [],
                   "predicted_fv_light_2": [], "sabdab_query_whole_2": [],
                   "sabdab_query_cdr_2": [], "sabdab_query_cdrh3_2": []}

train_ds_unbatched, train_ds_size = create_tf_dataset(
    "D:\PHD_CODE\csv_datasets\\" + "train_" + dataset_identity + ".csv")
val_ds_unbatched, val_ds_size = create_tf_dataset("D:\PHD_CODE\csv_datasets\\" + "val_" + "ds_18" + ".csv")
test_ds_unbatched, test_ds_size = create_tf_dataset("D:\PHD_CODE\csv_datasets\\" + "test_" + "ds_18" + ".csv")

vectorization_layer = Create_Vectorization_Layer()

train_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(train_ds_unbatched), size=train_ds_size)
val_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(val_ds_unbatched), size=val_ds_size)
test_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(test_ds_unbatched), size=test_ds_size)

all_probabilities = {}

# Loss and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
train_history = {}  # Epoch: loss, accuracy
val_history = {}  # Epoch: loss, accuracy
written_results["test_loss"] = []
written_results["test_accuracy"] = []

# Setting hyperparameters
vocab_size = vectorization_layer.vocab_size + 4
inp_seq_length = vectorization_layer.inp_seq_len + 2
out_seq_length = vectorization_layer.out_seq_len + 4

# Setting learning rate and optimiser
learning_rate = CustomSchedule(embedding_size)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Creating the Transformer model
transformer = Transformer(num_encoder_layers, num_decoder_layers, embedding_size,
                          num_heads, ffn_width,
                          vocab_size,
                          inp_seq_length,
                          out_seq_length,
                          dropout_rate)

temp_learning_rate_schedule = CustomSchedule(embedding_size)

early_prediction_made = False
extra_epoch = False

# Training, validation, and testing
for epoch in range(max(EPOCHS)):
    start = time.time()

    # Training
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_ds):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # Validation
    val_loss.reset_states()
    val_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(val_ds):
        validation_and_testing(inp, tar, "val")

    print('Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}'.format(
        epoch + 1, val_loss.result(), val_accuracy.result()))

    # Testing and predicting
    if epoch + 1 in EPOCHS:
        all_probabilities[epoch + 1] = {}
        transformer.save_weights(write_directory + "\\" + 'weights_epoch_' + str(epoch +1) + ".h5")
        test_loss.reset_states()
        test_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(test_ds):
            validation_and_testing(inp, tar, "test")

        print('Test Loss {:.4f} Test Accuracy {:.4f}'.format(test_loss.result(), test_accuracy.result()))
        written_results["test_loss"].append(str(test_loss.result().numpy()))
        written_results["test_accuracy"].append(str(test_accuracy.result().numpy()))
        predict_fv(antigen_sequence_1, epoch + 1, result_id=1)
    elif not early_prediction_made:
        if train_accuracy.result() > val_accuracy.result():
            all_probabilities[epoch + 1] = {}
            transformer.save_weights(write_directory + "\\" + 'weights_epoch_' + str(epoch + 1) + ".h5")
            test_loss.reset_states()
            test_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(test_ds):
                validation_and_testing(inp, tar, "test")

            print('Test Loss {:.4f} Test Accuracy {:.4f}'.format(test_loss.result(), test_accuracy.result()))
            written_results["test_loss"].append(str(test_loss.result().numpy()))
            written_results["test_accuracy"].append(str(test_accuracy.result().numpy()))
            predict_fv(antigen_sequence_1, epoch + 1, result_id=1)
            early_prediction_made = True
            extra_epoch = epoch + 1
    else:
        pass

    train_history[epoch + 1] = [train_loss.result().numpy(), train_accuracy.result().numpy()]
    val_history[epoch + 1] = [val_loss.result().numpy(), val_accuracy.result().numpy()]

plot_loss(train_history, val_history)
plot_accuracy(train_history, val_history)

# Storing information for writing to file
written_results["encoder_layers"] = str(num_encoder_layers)
written_results["decoder_layers"] = str(num_decoder_layers)
written_results["embedding_dim"] = str(embedding_size)
written_results["ffn_width"] = str(ffn_width)
written_results["num_heads"] = str(num_heads)
written_results["input_vocab_size"] = str(vocab_size)
written_results["target_vocab_size"] = str(vocab_size)
written_results["dropout_rate"] = str(dropout_rate)

if extra_epoch:
    EPOCHS = EPOCHS + [extra_epoch]
    EPOCHS = sorted(EPOCHS)

written_results["Epochs"] = EPOCHS
written_results["input_seq_length"] = str(inp_seq_length)
written_results["output_seq_length"] = str(out_seq_length)
written_results["antigen_sequence_1"] = antigen_sequence_1
written_results["dataset_identity"] = dataset_identity
written_results["schedule identity"] = schedule_identity
written_results["train_ds_size"] = str(train_ds_size)
written_results["val_ds_size"] = str(val_ds_size)
written_results["test_ds_size"] = str(test_ds_size)
written_results["model_identity"] = "Transformer_1"

Write_Results.write_results(write_directory, written_results, multiple_designs=False)
plt.close('all')

all_prob_file = open(write_directory + "\\" + "all_probabilities.txt", "w")
for key in all_probabilities.keys():
    all_prob_file.write("epoch " + str(key) + "\n")
    for key_2 in all_probabilities[key].keys():
        all_prob_file.write(str(all_probabilities[key][key_2]) + "\n")

    all_prob_file.write("\n")

