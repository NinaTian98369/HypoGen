#encoding:utf-8
import csv
import gc
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid   = guid 
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids   = input_ids  
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id

class CreateDataset(Dataset):
    def __init__(self,data,max_seq_len,tokenizer,example_type,seed):
        self.seed = seed
        self.data = data
        self.max_seq_len  = max_seq_len
        self.example_type = example_type
        self.tokenizer = tokenizer
        self.build_examples()


    def read_data(self,data_path,quotechar = None):
        lines = []
        with open(data_path,'r',encoding='utf-8') as fr:
            reader = csv.reader(fr,delimiter = '\t',quotechar = quotechar)
            for line in reader:
                lines.append(line)
        return lines

    def build_examples(self):
        if isinstance(self.data,Path):
            lines = self.read_data(data_path=self.data)
        else:
            lines = self.data
        self.examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(self.example_type,i)
            text_a = line[0]
            label = line[1]
            if isinstance(label,str):
                label = [np.float32(x) for x in label.split(",")]
            else:
                label = [np.float32(x) for x in list(label)]
            text_b = None
            example = InputExample(guid = guid,text_a = text_a,text_b=text_b,label= label)
            self.examples.append(example)
        del lines
        del self.data
        gc.collect()

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def build_features(self,example):
        '''
        # E.g.1
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # E.g.2
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        '''
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self.truncate_seq_pair(tokens_a,tokens_b,max_length = self.max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with '-2'
            if len(tokens_a) > self.max_seq_len - 2:
                tokens_a = tokens_a[:self.max_seq_len - 2]
        

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)  # 对应type_ids

        
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)

        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1] * len(input_ids)
        # zero padding
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        # label
        label_id = example.label
        feature = InputFeature(input_ids = input_ids,input_mask = input_mask,
                               segment_ids = segment_ids,label_id = label_id)
        return feature

    def preprocess(self,index):
        example = self.examples[index]
        feature = self.build_features(example)
        return np.array(feature.input_ids),np.array(feature.input_mask),\
               np.array(feature.segment_ids),np.array(feature.label_id)

    def __getitem__(self, index):
        return self.preprocess(index)

    def __len__(self):
        return len(self.examples)