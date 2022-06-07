import create_tfrecords
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import sentencepiece as spm
from tensorflow import keras
from itertools import cycle
from functools import partial


@tf.function
def decode_example(ex, spm_eng, spm_fra):
    feature_description = {
        "input_text": tf.io.FixedLenFeature([], tf.string),
        "target_text": tf.io.FixedLenFeature([], tf.string)
    }
    decoded = tf.io.parse_single_example(ex, feature_description)
    encoder_in = spm_eng.tokenize(decoded["input_text"])
    decoder_in = spm_fra.tokenize(decoded["target_text"])
    return ({
                "encoder_in": encoder_in,
                "decoder_in": decoder_in[:-1],
            },
            {
                "target": decoder_in[1:]
            })


def get_dataset(spm_eng, spm_fra):
    files = tf.data.Dataset.list_files('output/fra-data/train-?????.tfrecords')
    shards = files.shuffle(1000).repeat()
    decoder = partial(decode_example, spm_eng=spm_eng, spm_fra=spm_fra)
    #data = shards.interleave(tf.data.TFRecordDataset, cycle_length=4).shuffle(buffer_size=8192).map(decoder).padded_batch(batch_size)
    data = (shards
            .interleave(tf.data.TFRecordDataset, cycle_length=4)
            .shuffle(buffer_size=8192)
            .map(decoder)
            .padded_batch(batch_size)
            )
    return data


def build_model(
        num_encoder_tokens,
        num_decoder_tokens,
        embedding_dim,
        latent_dim,
):
    encoder_inputs = keras.Input(shape=(None,), name="encoder_in")
    encoder_embedding = keras.layers.Embedding(
        num_encoder_tokens, embedding_dim, mask_zero=True, name="encoder_embedding"
    )(encoder_inputs)

    encoder_rnn = keras.layers.LSTM(
        latent_dim, return_state=True, return_sequences=True, name="encoder_rnn"
    )
    encoder_outputs, state_h, state_c = encoder_rnn(encoder_embedding)

    decoder_inputs = keras.Input(shape=(None,), name="decoder_in")
    decoder_embedding = keras.layers.Embedding(
        num_decoder_tokens, embedding_dim, mask_zero=True, name="decoder_embedding"
    )(decoder_inputs)

    decoder_rnn = keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_rnn"
    )
    decoder_rnn_output, _, _ = decoder_rnn(decoder_embedding, initial_state=[state_h, state_c])

    att_queries = keras.layers.Dense(latent_dim, use_bias=False, name="Wq")(decoder_rnn_output)
    att_values = keras.layers.Dense(latent_dim, use_bias=False, name="Wv")(encoder_outputs)
    att_keys = keras.layers.Dense(latent_dim, use_bias=False, name="Wk")(encoder_outputs)

    context_vector, att_weights = keras.layers.Attention()(
        inputs=[att_queries, att_values, att_keys],
        return_attention_scores=True
    )

    concat = keras.layers.Concatenate(axis=-1)([context_vector, decoder_rnn_output])
    attention = keras.layers.Dense(latent_dim, use_bias=False, name="Wc")(concat)

    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name="classifier")
    decoder_outputs = keras.layers.TimeDistributed(decoder_dense, name="target")(attention)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


if __name__ == "__main__":
    with open("output/eng.model", "rb") as f:
        eng_model_proto = f.read()
        english = text.SentencepieceTokenizer(model=eng_model_proto, add_eos=True)
    with open("output/fra.model", "rb") as f:
        fra_model_proto = f.read()
        french = text.SentencepieceTokenizer(model=fra_model_proto, add_bos=True, add_eos=True)

    batch_size = 32
    epochs = 10
    embedding_dim = 32
    latent_dim = 512
    fc_dim = 2 * latent_dim
    num_samples = None
    num_encoder_tokens = 10000
    num_decoder_tokens = 10000

    model = build_model(
        num_encoder_tokens, num_decoder_tokens, embedding_dim, latent_dim
    )

    print(model.summary())

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    data = get_dataset(english, french)
    model.fit(
        data,
        epochs=1,  # 10,
        steps_per_epoch=100,  #6000,
    )
    model_save_file = "output/model/"
    model.save(model_save_file)
