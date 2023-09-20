import os
import tensorflow as tf
from BertLibrary.bert_predictor import BertPredictor
from BertLibrary.bert_trainer import BertTrainer
from BertLibrary.bert_evaluator import BertEvaluator

from tensorflow.estimator import Estimator
from tensorflow.estimator import RunConfig

from BertLibrary.bert.run_classifier import