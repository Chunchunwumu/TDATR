from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import os
from TDATR_utils.dataclass import HulkDataclass
import re
import numpy as np

import torch
from torch import Tensor
import sentencepiece as spm
import logging
import json

logger= logging.getLogger(__name__)




#-------------------------------------------------------------------------------
def load_json(filename):
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    #print("load json from: {}".format(filename))
    return data



#-------------------------------------------------------------------------------
def save_json(data, file_path):
    dst_dir = os.path.dirname(file_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    print("save json to: ", file_path)



@dataclass
class SPTokenizerConfig(HulkDataclass):
    vocab_file: str = field(
        default="", metadata={"help": "path to prefix for sentencepiece model(vocab) file"}
    )

class BaseTokenizer(object):
    def __len__(self):
        raise NotImplementedError

    @property
    def eod(self):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError



class SPTokenizer(object):
    def __init__(self, cfg):
        model_file = cfg.vocab_file + ".model"
        vocab_file = cfg.vocab_file + ".vocab"
        logger.info("\n\nSPTokenizer_vocab_file:{}".format(cfg.vocab_file))
        
        assert os.path.exists(vocab_file), \
                f"vocab file path ({vocab_file}) is not exist"
        assert os.path.exists(model_file), \
                f"sentencepiece model path ({model_file}) is not exist"
        f = open(vocab_file,'r', encoding='utf-8')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]
        f.close()
        self.decoder = {v:k for k,v in self.encoder.items()}
        
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans("\t\n", "  ")
        self.sep_id = self.encoder['<s>']  # 1
        self.eod_id = self.encoder['<end>']
        self.pad_id = self.encoder['<pad>']
        self.unk_id = self.encoder["<unk>"]
        
    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder) 

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        text= text.translate(self.translator)
        return self.sp.encode(text)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        return text

    def tokenize_sp(self, text):
        sp_list = [k for k in self.encoder.keys() if "iflytek" in k]
        sp_text = "|".join(sp_list)
        sp_text = "({})".format(sp_text)
        text_list = re.split(sp_text, text)
        res = []
        for utter in text_list:
            if len(utter):
                if utter == "<iflytek_s>":
                    res.append(self.sep_id)
                elif utter == "<iflytek_end>":
                    res.append(self.eod_id)
                else:
                    res += self.sp.encode(utter)
        return res


    def get_sp_style(self):
        '''
        sp_list = []
        for i in range(1000):
            x = '<pos_x_{}>'.format(i)
            y = '<pos_y_{}>'.format(i)
            sp_list.append(x)
            sp_list.append(y)
        sp_list.extend([x, y])
        sp_list.extend(['<box_s>', '<box_e>'])

        self.sp_token_file = os.path.join('/train28/mmu/permanent/ypcui/code/kosmos2.5/910b_v1/kosmos2.5_910b_20240124/kosmos2.5_910b/minigpt4/tokenizers', os.path.basename(self.sp_token_file))
        save_json(sp_list, self.sp_token_file)
        for sp in sp_list:
            assert sp in self.encoder
        print(sp_list)
        '''

        sp_list = load_json(self.sp_token_file)
        sp_list.append(self.bos_token)
        sp_list.append(self.eos_token)
        text = "|".join(sp_list)
        text = "({})".format(text)
        return text
    