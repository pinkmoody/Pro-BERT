
from BertLibrary.bert.run_classifier import *
import BertLibrary.bert.modeling as modeling

from .BertModel import BertModel
from BertLibrary.processors.FEProcessor import FEProcessor

from tensorflow.estimator import EstimatorSpec

import sys
import os
import tensorflow as tf


class BertFEModel(BertModel):

    def __init__(self,
                 model_dir,
                 ckpt_name,
                 layer,
                 do_lower_case,
                 max_seq_len,
                 batch_size,
                 config=None):
        super().__init__(
            model_dir=model_dir,
            ckpt_name=ckpt_name,
            do_lower_case=do_lower_case,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            labels=[0],
            config=config,
            trainable=False)

        model_fn_args = {'bert_config': self.bert_config,
                         'layer': layer,
                         'init_checkpoint': self.init_checkpoint}
        config_args = {}

        self.build(model_fn_args, config_args)

        self.processer = FEProcessor(
            max_seq_len, self.tokenizer, batch_size, pred_key='predictions')

    def get_model_fn(self,
                     bert_config,
                     layer,
                     init_checkpoint):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" %
                                (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
