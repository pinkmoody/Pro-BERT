import os
import tensorflow as tf
from BertLibrary.bert_predictor import BertPredictor
from BertLibrary.bert_trainer import BertTrainer
from BertLibrary.bert_evaluator import BertEvaluator

from tensorflow.estimator import Estimator
from tensorflow.estimator import RunConfig

from BertLibrary.bert.run_classifier import *
import BertLibrary.bert.modeling as modeling
import BertLibrary.bert.tokenization as tokenization


class BertModel:

    def __init__(self,
                 model_dir,
                 ckpt_name,
                 do_lower_case,
                 max_seq_len,
                 batch_s