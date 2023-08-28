import tensorflow as tf
import math
import numpy as np


class BertPredictor:

    def __init__(self, model_estimator, processor, config):
        self.processor = processor
        self.predictor = tf.contrib.predictor.from_estimator(
            model_estimator, processor.serving_input_receiver_fn(), config=config)
        self.bs = self.processor.batch_size

    def __call__(self, sentences):
    