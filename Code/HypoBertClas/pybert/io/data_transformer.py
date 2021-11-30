#encoding:utf-8
import random
import operator
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ..utils.utils import text_write
from ..utils.utils import pkl_write
import numpy as np

class DataTransformer(object):
    def __init__(self,
                 logger,
                 seed,
                 add_unk = True
                 ):
        self.seed          = seed
        self.logger        = logger
        self.item2idx = {}
        self.idx2item = []
        #
        if add_unk:
            self.add_item('<unk>')

    def add_item(self,item):
        '''
        add new item to dict
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1

    def get_idx_for_item(self,item):
        '''
        return 0 for unk
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return 0

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode('UTF-8')

    def get_items(self):
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))

    def split_sent(self,line):
        res = line.strip('\n').split()
        return res

    def train_val_split(self,X, y,valid_size,
                        stratify=True,
                        shuffle=True,
                        save = True,
                        train_path = None,
                        valid_path = None,
                        dev_path = None,
                        upsample_rate = 2):
        shuffle = False
        upsample = True
        self.logger.info('train val split')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
                #self.logger.info(str(data_y))
                bucket[int(data_y)].append((data_x, data_y))
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(self.seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size/2])
                train.extend(bt[test_size:])
            
            if shuffle:
                random.seed(self.seed)
                random.shuffle(train)
        else:
            data = []
            for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
                #print(data_y)
                data.append((data_x, data_y))
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                print('shuffle')
                random.seed(self.seed)
                random.shuffle(data)
            else:
                print('Validate on hypo-en only...')
            valid = data[:int(test_size)]
            train = data[test_size:]

            print('Before upsample: ',len(train))
            
            if upsample:
                for i, t in enumerate(train[:len(train)]):
                    if t[1][1] ==  1:
                        train.append(t)
                    if t[1][1] != 1 and t[1][1] != 0:
                        print('error!')
                print('After upsample: ',len(train))
            
            
            if shuffle:
                random.seed(self.seed)
                random.shuffle(train)

        if save:
            text_write(filename=train_path, data=train)
            text_write(filename=valid_path, data=valid)
        return train, valid

    def build_vocab(self,data,min_freq,max_features,save,vocab_path):
        '''
        :param data:
        :param min_freq:
        :param max_features:
        :param save:
        :param vocab_path:
        :return:
        '''
        count = Counter()
        self.logger.info('Building word vocab')
        for i,line in enumerate(data):
            words = self.split_sent(line)
            count.update(words)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1))
        # dict
        all_words = [w[0] for w in count if w[1] >= min_freq]
        if max_features:
            all_words = all_words[:max_features]

        self.logger.info('vocab_size is %d' % len(all_words))
        for word in all_words:
            self.add_item(item = word)
        if save:
            pkl_write(data = self.item2idx,filename = vocab_path)

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences, ids = [], [], []
        data = pd.read_csv(raw_data_path,delimiter = '\t',error_bad_lines = False)
        for row in tqdm(data.values):
            if is_train:
                target = row[2:]
            else:
                target = [-1,-1,-1,-1,-1,-1]
            sentence = str(row[1])
            # preprocess
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
            if is_train == False:
                ids.append(row[0])
        return targets,sentences,ids
