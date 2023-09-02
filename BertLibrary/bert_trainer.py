from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *


class BertTrainer():

    def __init__(self, model):
        self.model = model
        self.processor = Label2TextPr