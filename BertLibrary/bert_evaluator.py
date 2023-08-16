from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *

class BertEvaluator:

    def __init__(self, model, iter_steps=1000):
        self.model = model
        self.processor = Label2TextProcessor(self.model.max_seq_len)
        self.logging_hook = LoggingSessionHook(self.model, iter_steps)

    def convert_features(self, data_path, output_file):
        examples = self.processor.file_get_examples(data_path)
        file_based_convert_examples_to_features(examples,
                                                self.model.labels,
                                                self.model.max_seq_len,
                                                self.model.tokenizer,
                                                output_file)

    def evaluate(self, X, y, checkpoint=None):
        test_examples = self.processor.get_examples(X, y)

        test_features = convert_examples_to_features(test_examples,
            self.model.labels,
            self.model.max_seq_len,
            self.model.tokenizer)

        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=self.model.max_seq_len,
            is_training=False,
            drop_remainder=False)

        self.model.estimator.evaluate(
          test_input_fn, checkpoint_path=checkpoint, hooks=[self.logging_hook])

    def evaluate_from_file(self, data_path, checkpoint=None):
        test_file = os.path.join(data