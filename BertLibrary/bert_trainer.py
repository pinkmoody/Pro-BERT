from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *


class BertTrainer():

    def __init__(self, model):
        self.model = model
        self.processor = Label2TextProcessor(self.model.max_seq_len)

    def convert_features(self, data_path, output_file):
        examples = self.processor.file_get_examples(data_path)
        file_based_convert_examples_to_features(examples,
                                                self.model.labels,
                                                self.model.max_seq_len,
                                                self.model.tokenizer,
                                                output_file)

    def train(self, X, y, steps, X_val=None, y_val=None, eval_cooldown=600):
        train_examples = self.processor.get_examples(X, y)

        train_features = convert_examples_to_features(train_examples,
            self.model.labels,
            self.model.max_seq_len,
            self.model.tokenizer)

        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=False)

        if X_val and y_val:
            dev_examples = self.