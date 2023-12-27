
# Pro-BERT\n\nPro-BERT, a Tensorflow library, provides the simplicity and speed required for quick and efficient model training and finetuning based on Bert.\n\n**\*\*\*\*\*\* New May 31st, 2019: Whole Word Masking Models \*\*\*\*\*\*\*\n\nIn the original pre-processing code, random WordPiece tokens are selected for masking. But the new method, called Whole Word Masking, masks all the tokens corresponding to a word at once. Beneficially, it maintains the same overall masking rate.\n\nThis library also includes fully detailed content on different models and their various use cases, providing a comprehensive overview for users to idenfity the best model that suits their need. Also, a clear and straight forward guide on how to run the `run_classifier.py` script to perform sentence (and sentence-pair) classification tasks is provided. In addition, an example code is included to perform fine-tuning of the `BERT-Base` on the Microsoft Research Paraphrase Corpus (MRPC) corpus.\n\nThis code works comfortably with a CPU, GPU, and Cloud TPU. However, users are expected to encounter out-of-memory issues on GPUs due to the fact that all experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of device RAM. Consequently, when using a GPU with 12GB - 16GB of RAM, you are likely to encounter out-of-memory issues if you use the same hyperparameters described in the paper.\n\nNote: You are advised to add a slight amount of noise to your input data (e.g., randomly truncate 2% of input segments) to make it more robust to non-sentential input during fine-tuning.