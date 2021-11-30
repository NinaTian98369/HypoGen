#encoding:utf-8
import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from collections import OrderedDict

def prepare_device(n_gpu_use,logger):
    """
    setup GPU device if available, move model into configured device
     """
    if isinstance(n_gpu_use,int):
        n_gpu_use = range(n_gpu_use)
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = range(0)
    if len(n_gpu_use) > n_gpu:
        msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device('cuda:%d'%n_gpu_use[0] if len(n_gpu_use) > 0 else 'cpu')
    list_ids = n_gpu_use
    return device, list_ids

def model_device(n_gpu,model,logger):
    '''
    :param n_gpu:
    :param model:
    :param logger:
    :return:
    '''
    device, device_ids = prepare_device(n_gpu,logger)
    if len(device_ids) > 1:
        logger.info("current {} GPUs".format(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model,device

def restore_checkpoint(resume_path,model = None,optimizer = None):
    '''
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    '''
    if isinstance(resume_path,Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return [model,optimizer,best,start_epoch]


def load_bert(model_path,model = None,optimizer = None):
    '''
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    '''
    if isinstance(model_path,Path):
        model_path = str(model_path)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # new_state_dict = {}
    # for key,value in state_dict.items():
        # if "module" in key:
        #     new_state_dict[key.replace("module.","")] = value
        # else:
        #     new_state_dict[key] = value
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return [model,optimizer,best,start_epoch]

def seed_everything(seed = 1029,device='cpu'):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    '''
    batch
    :param batch:
    :return:
    '''
    r"""Puts each data field into a tensor with outer dimension batch size"""
    transposed = zip(*batch)
    lbd = lambda batch:torch.cat([torch.from_numpy(b).long() for b in batch])
    return [lbd(samples) for samples in transposed]

class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val  = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def summary(model, *inputs, batch_size=-1, show_input=True):

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def json_write(data,filename):
    with open(filename,'w') as f:
        json.dump(data,f)

def json_read(filename):
    with open(filename,'r',encoding="utf-8") as f:
        return json.load(f)

def pkl_read(filename):
    with open(filename,'rb',encoding="utf-8") as f:
        return pickle.load(f)

def pkl_write(filename,data):
    with open(filename, 'wb',encoding="utf-8") as f:
        pickle.dump(data, f)

def text_write(filename,data):
    with open(filename,'w',encoding = 'utf-8') as fw:
        for sentence,target in tqdm(data,desc = 'write data to disk'):
            target  = [str(x) for x in target]

            line = '\t'.join([sentence,",".join(target)])
            #print(line)
            fw.write(line.encode('utf-8').decode("utf-8") )
            fw.write('\n')