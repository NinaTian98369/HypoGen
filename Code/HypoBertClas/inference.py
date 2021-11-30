#encoding:utf-8
import torch
import warnings
from torch.utils.data import DataLoader
from pybert.io.dataset import CreateDataset
from pybert.io.data_transformer import DataTransformer
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything
from pybert.config.basic_config import configs as config
from pybert.model.nn.bert_fine import BertFine
from pybert.test.predicter import Predicter
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer
warnings.filterwarnings("ignore")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # **************************** log ***********************
    logger = init_logger(log_name=config['model']['arch'], log_dir=config['output']['log_dir'])
    logger.info(f"seed is {config['train']['seed']}")
    device = 'cuda:%d' % config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'
    seed_everything(seed=config['train']['seed'],device=device)
    logger.info('starting load data from disk')
    id2label = {value: key for key, value in config['label2id'].items()}
    #**************************** data ***********************
    DT = DataTransformer(logger = logger,seed = config['train']['seed'])


    targets, sentences,ids = DT.read_data(raw_data_path=config['data']['test_file_path'],
                                      preprocessor=EnglishPreProcessor(),
                                      is_train=False)
    tokenizer = BertTokenizer(vocab_file=config['pretrained']['bert']['vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    # test
    test_dataset   = CreateDataset(data  = list(zip(sentences,targets)),
                                   tokenizer = tokenizer,
                                   max_seq_len = config['train']['max_seq_len'],
                                   seed = config['train']['seed'],
                                   example_type = 'test')
    test_loader = DataLoader(dataset     = test_dataset,
                             batch_size  = config['train']['batch_size'],
                             num_workers = config['train']['num_workers'],
                             shuffle     = False,
                             drop_last   = False,
                             pin_memory  = False)

  
    # **************************** load trained model ***********************
    logger.info("initializing model")
    model = BertFine.from_pretrained(config['pretrained']['bert']['bert_model_dir'],
                                     cache_dir=config['output']['cache_dir'],
                                     num_classes = len(id2label))

    logger.info('model predicting....')
    predicter = Predicter(model = model,
                         logger = logger,
                         n_gpu=config['train']['n_gpu'],
                         model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                         #model_path = config['output']['checkpoint_dir'] / f"epoch_1_0.5206_bert_model.pth",
                         )
    # predict
    result = predicter.predict(data = test_loader)

    wfname = config['data']['test_file_path']
    print('Saving to ' + 'prediction/prediction_' + wfname)
    file = open('prediction/prediction_' + wfname, 'w')

    for index, line, score in zip(ids, sentences, result):
        file.write(str(index) + '\t' + line + '\t' + str(score[1]))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    main()
