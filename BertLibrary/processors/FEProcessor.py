import tensorflow as tf
from .Processor import Processor

class FEProcessor(Processor):

    def __init__(self, max_seq_len, tokenizer, batch_size, pred_key):
        super().__init__(max_seq_len, tokenizer, batch_size, pred_key)

    def serving_input_receiver_fn(self):
        features = {
            "input_ids": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='input_ids'),
            "input_mask": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='input_mask'),
            "segment_ids": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='segment_ids'),
        }

        return tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    def preprocess_sentences(self, sentences):
        features = {'input_ids': [],
                    'input_mask': [],
                    'se