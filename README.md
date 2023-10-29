
# Bert as a Library (BaaL)
  
![GitHub](https://img.shields.io/github/license/kpi6research/Bert-as-a-Library)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/BertLibrary)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kpi6research/Bert-as-a-Library/blob/master/examples/Finetune_Bert_Sentiment140_with_BertLibrary.ipynb)
<a href="https://nbviewer.jupyter.org/github/kpi6research/Bert-as-a-Library/blob/master/examples/Finetune_Bert_Sentiment140_with_BertLibrary.ipynb" 
   target="_new">
   <img  
      src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg?sanitize=true" 
      width="109" height="20">
</a>


Bert as a Library is a framework for prediction, evaluation and finetuning of Bert models. It's also suitable for production, allowing an easy deploy using Flask or similar services.

## Installation
You can install the library from `pip` using following command
```bash
$ pip install BertLibrary
```

## Setup
The setup of BertLibrary is dead simple. You have 2 options:
- `import BertFTModel` to finetune your model and run evaluations/predictions.
- `import BertFEModel` if you want to extract features from a pretrained/finetuned Bert (only prediction/evaluation).
- Download a pretrained model from [here](https://github.com/google-research/bert)

### Finetuning Model
```python
# Import model for fintuning
from BertLibary import BertFTModel

# Instantiate the model
ft_model = BertFTModel( model_dir='uncased_L-12_H-768_A-12',
                        ckpt_name="bert_model.ckpt",
                        labels=['0','1'],
                        lr=1e-05,
                        num_train_steps=30000,
                        num_warmup_steps=1000,
                        ckpt_output_dir='output',
                        save_check_steps=1000,
                        do_lower_case=False,
                        max_seq_len=50,
                        batch_size=32,
                      )


ft_trainer =  ft_model.get_trainer()
ft_predictor =  ft_model.get_predictor()
ft_evaluator = ft_model.get_evaluator()

```

BertFTModel constructor parameters:

| Command | Description |
| ------ | ------ |
| model_dir | The path to the Bert pretrained model directory  |
| ckpt_name | The name of the checkpoint you want use |
| labels | The list of unique label names (must be string) |
| lr | The learning rate you will use during the finetuning |
| num_train_steps | The default number of steps to run the finetuning if not specified |
| num_warmup_steps | Number of warmup steps, see the original paper for more |
| ckpt_output_dir | The directory to save the finetuned model checkpoints |
| save_check_steps | Save and evaluate the model every save_check_steps |
| do_lower_case | Do a lower case during preprocessing if set to true |
| max_seq_len | Set a max sequence length of the model (max 512) |
| batch_size | Regulate the batch size for training/evaluation/prediction |
| config | (Optional) Tensorflow config object |


### Feature Extraction Model
```python