import sentencepiece as spm
import tensorflow as tf

from itertools import cycle


def read_input_and_target_texts(fname, num_samples=None):
    input_texts = []
    target_texts = []

    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    if num_samples:
        sample_rate = num_samples/len(lines)

    for line in lines:
        if num_samples and np.random.rand() > sample_rate:
            continue
        try:
            input_text, target_text, _ = line.split("\t")
            input_text = input_text
            target_text = target_text
            input_texts.append(input_text)
            target_texts.append(target_text)
        except:
            pass

    return input_texts, target_texts


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _string_feature(value):
    return _bytes_feature(value.encode("utf-8"))


def encode_example(input_text, target_text):
    features = {
        "input_text": _string_feature(input_text),
        "target_text": _string_feature(target_text)
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def write_tf_to_files(input_texts, target_texts, num_files):
    writers = []
    for file_num in range(num_files):
        filename = f"output/fra-data/train-{file_num:05d}.tfrecords"
        writer = tf.io.TFRecordWriter(filename)
        writers.append(writer)

    for input_text, target_text, writer in zip(input_texts, target_texts, cycle(writers)):
        encoded_example = encode_example(input_text, target_text)
        writer.write(encoded_example)

    for writer in writers:
        writer.close()


if __name__ == "__main__":
    data_path = "data/fra.txt"
    num_encoder_tokens = 10000
    num_decoder_tokens = 10000

    input_texts, target_texts = read_input_and_target_texts(data_path)
    with open("output/eng_sentences.txt", "w") as f:
        f.write("\n".join(input_texts))
        f.write("\n")
    with open("output/fra_sentences.txt", "w") as f:
        f.write("\n".join(target_texts))
        f.write("\n")
    spm.SentencePieceTrainer.train(
        input="output/eng_sentences.txt",
        model_prefix="output/eng",
        vocab_size=num_encoder_tokens,
        character_coverage=1.0,
        model_type="unigram",
        add_dummy_prefix=False,
        eos_id=3,
        bos_id=2,
        unk_id=1,
        pad_id=0
    )
    spm.SentencePieceTrainer.train(
        input="output/fra_sentences.txt",
        model_prefix="output/fra",
        vocab_size=num_decoder_tokens,
        character_coverage=1.0,
        model_type="unigram",
        add_dummy_prefix=False,
        eos_id=3,
        bos_id=2,
        unk_id=1,
        pad_id=0
    )
    write_tf_to_files(input_texts, target_texts, 5)
