#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path
"""Note:
please adapt the path
"""
BASE_DIR = Path('pybert')

configs = {

    'task':'multi label',

    'data':{
        'raw_data_path': 'path_to/HypoBertClas/pybert/dataset/raw/trainClas_all_clean.txt',
        'train_file_path': 'path_to/HypoBertClas/pybert/dataset/processed/train.tsv',
        'valid_file_path': 'path_to/HypoBertClas/pybert/dataset/processed/valid.tsv',
        'dev_file_path': 'path_to/HypoBertClas/pybert/dataset/processed/dev.tsv',
        'test_file_path': 'pybert/dataset/inference/inference_0426.txt'
    },
    'output':{
        'log_dir': 'path_to/HypoBertClas/pybert/output/log', 
        'writer_dir': "path_to/HypoBertClas/pybert/output/TSboard",
        'figure_dir': "path_to/HypoBertClas/pybert/output/figure", 
        'checkpoint_dir': "path_to/HypoBertClas/pybert/output/checkpoints",
        'cache_dir': 'path_to/HypoBertClas/pybert/model/',
        'result': "path_to/HypoBertClas/pybert/output/result",
    },
    'pretrained':{
        "bert":{
            'vocab_path':   'path_to/HypoBertClas/pybert/model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
            'tf_checkpoint_path':  'path_to/HypoBertClas/pybert/model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
            'bert_config_file': 'path_to/HypoBertClas/pybert/model/pretrain/uncased_L-12_H-768_A-12/bert_config.json',
            'pytorch_model_path': 'path_to/HypoBertClas/pybert/model/pretrain/pytorch_pretrain/pytorch_model.bin',
            'bert_model_dir':  'path_to/HypoBertClas/pybert/model/pretrain/pytorch_pretrain',
        },
        'embedding':{}
    },
    'train':{
        'valid_size': 0.1,
        'max_seq_len': 40,
        'do_lower_case':True,
        'batch_size': 32,
        'epochs': 5,  
        'start_epoch': 1,
        'warmup_proportion': 0.1,# Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        'gradient_accumulation_steps': 1,# Number of updates steps to accumulate before performing a backward/update pass.
        'learning_rate': 2e-5,
        'n_gpu': [0,1,2,3],
        'num_workers': multiprocessing.cpu_count(),
        'weight_decay': 1e-5,
        'seed':2021,
        'resume':False,
    },
    'predict':{
        'batch_size':200,
    },
    'callbacks':{
        'lr_patience': 2, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',  
        'early_patience': 20,   # early_stopping
        'save_best_only': True,
        'save_checkpoint_freq': 10 
    },
    'label2id' : { # for binary classification
        "positive": 0,
        "negative": 1,
    },
    'model':{
        'arch':'bert'
    }
}