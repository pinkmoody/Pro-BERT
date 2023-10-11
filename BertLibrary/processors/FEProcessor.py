import tensorflow as tf
from .Processor import Processor

class FEProcessor(Processor):

    def __init__(self, max_seq_len, tokenizer, batch_size, pred_key):
        super().__init__(max_seq_len, tokenizer, batch_size, pred_key)

    def serving_input_receiver_fn(self):
        features = {
            "input_