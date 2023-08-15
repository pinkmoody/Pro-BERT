from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *

class BertEvaluator:

    def __init__(self, model, iter_steps=1000):
        self.model = model
        self.processor = Label2TextProcessor(self.model.max_seq_len)
        self.logging_hook = LoggingSessionHook(self.model, iter_s