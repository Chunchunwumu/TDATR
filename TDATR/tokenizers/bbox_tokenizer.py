import os
import re
import logging

from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, List
import numpy as np
import sentencepiece as spm

import torch
from torch import Tensor

import json

from TDATR_utils.dataclass import HulkDataclass
from TDATR.tokenizers.sp_tokenizer import SPTokenizerConfig, SPMiniGPT4Tokenizer

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
class BboxTokenConfig(SPTokenizerConfig):
    patch_num: Optional[int] = field(
        default= 32,
        metadata={"help": "the patch num"}
    )
    prefix_flag: Optional[str] = field(
        default= "<cap_with_bbox>",
        metadata={"help": "distinguish the caption with bbox and the normal caption"}
    )
    grounding_token: Optional[str] = field(
        default= "<grounding>",
        metadata={"help": "grounding token"}
    )
    phrase_bos_token: Optional[str] = field(
        default= "<phrase>",
        metadata={"help": "phrase bos token"}
    )
    phrase_eos_token: Optional[str] = field(
        default= "</phrase>",
        metadata={"help": "phrase eos token"}
    )
    object_bos_token: Optional[str] = field(
        default= "<object>",
        metadata={"help": "object bos token"}
    )
    object_eos_token: Optional[str] = field(
        default= "</object>",
        metadata={"help": "object eos token"}
    )
    text_bos_token: Optional[str] = field(
        default= "<ifly_text>",
        metadata={"help": "text bos token"}
    )
    text_eos_token: Optional[str] = field(
        default= "</ifly_text>",
        metadata={"help": "text eos token"}
    )
    multi_object_delimiter_token: Optional[str] = field(
        default= "<multi_object_delimiter>",
        metadata={"help": "multi object delimiter token"}
    )
    patch_token: Optional[str] = field(
        default= "<patch_index_#>",
        metadata={"help": "patch token prefix"}
    )


class BboxTokenizer(SPMiniGPT4Tokenizer):
    def __init__(self, cfg: BboxTokenConfig):
        super().__init__(cfg)
        self.patch_num = cfg.patch_num
        self.pflg = cfg.prefix_flag

        
        self.tbos = cfg.text_bos_token
        self.teos = cfg.text_eos_token

        self.sp_list = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tokenizer.sp_token_list.json"))
        
        self.added_tokens = []
        self.added_tokens.extend(self.sp_list) 
       
        self.added_vocab_size = len(self.added_tokens)
        self.origin_vocab_size = super().vocab_size
    
        self.ocr_vocab_size = 2
        self.ocr_tokens = [self.tbos, self.teos]
   

        for i, token in enumerate(self.added_tokens):
            self.encoder[token] = i + self.origin_vocab_size
            self.decoder[i + self.origin_vocab_size] = token
        
        for i, token in enumerate(self.ocr_tokens):
            self.encoder[token] = i + self.origin_vocab_size + self.added_vocab_size
            self.decoder[i + self.origin_vocab_size + self.added_vocab_size] = token

        
        from TDATR_utils.global_context import global_context as gpc    
        self.bos_token = gpc.config.model.bos_token
        self.eos_token = gpc.config.model.eos_token
        self.pad_token = gpc.config.model.pad_token
    
        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]
        self.pad_token_id = self.encoder[self.pad_token]
        
        # self.gtoken_id = self.encoder[self.gtoken]
        # self.oeos_id = self.encoder[self.oeos]
        

        text = "\n\n#--------------------bbox_tokenizer_minigpt4 token info------------------------"
        text += '\n origin_vocab_size:{}'.format(self.origin_vocab_size)
        text += '\n add token:{}'.format(len(self.added_tokens))
        text += '\n ocr token:{}'.format(len(self.ocr_tokens))
        text += '\n all token:{}\n\n'.format(len(self.encoder))
        logger.info(text)
        assert len(self.encoder)==len(self.decoder)


        
        self.cell_s_id_1 = self.tokenize_sp("<iflytek_html_td_s>")[0]
        self.cell_s_id_2 = self.tokenize_sp("<iflytek_html_span_s>")[0]
        self.cell_e_id = self.tokenize_sp("<iflytek_html_td_e>")[0]
        self.row_s_id = self.tokenize_sp("<iflytek_html_tr_s>")[0]
        self.row_e_id = self.tokenize_sp("<iflytek_html_tr_e>")[0]
        self.col_span_id = self.tokenize_sp("<iflytek_html_colspan>")[0]
        self.row_span_id = self.tokenize_sp("<iflytek_html_rowspan>")[0]
        self.span_e = self.tokenize_sp("<iflytek_html_span_e>")[0]

        
    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder)
    
    def detokenize(self, tokens):
        
        chunk = []
        res = ''
        for e in tokens:
            if e >= self.origin_vocab_size:
                if len(chunk) > 0:
                    res += super().detokenize(chunk)
                    chunk = []
                res += self.decoder[e]
            else:
                chunk.append(e)
        if len(chunk) > 0:
            res += super().detokenize(chunk)
        return res

    def tokenize_sp(self, text):
        

        sp_list = [k for k in self.encoder.keys() if "iflytek" in k]
        sp_tokens = self.added_tokens + self.ocr_tokens + sp_list
        text_list = re.split(f"({'|'.join(sp_tokens)})", text)
        res = []
        for utter in text_list:
            if len(utter):
                if utter in sp_tokens:
                    res.append(self.encoder[utter])
                else:
                    res += super().tokenize_sp(utter)
        return res

    
    def call_encoder_cell_ranges(
            self,
            texts: List[str],
            max_length: Optional[int] = None,
            device: Optional[torch.device] = torch.device("cpu"),
            bias=0,
            
    ): # 在解码的基础上找出单元格范围
        batch_tokens = []
        batch_attention_mask = []
        cur_batch_max_len = 0
        batch_ranges = list()

        for text in texts:
            tokens = self.tokenize_sp(text)
            row_start_ids = np.where(np.array(tokens)==self.row_s_id)[0]
            row_end_ids = np.where(np.array(tokens)==self.row_e_id)[0]
            cell_start_ids_1 = np.where(np.array(tokens)==self.cell_s_id_1)[0]
            cell_start_ids_2 = np.where(np.array(tokens)==self.cell_s_id_2)[0]
            # print(cell_start_ids_1, cell_start_ids_2)
            cell_start_ids = np.concatenate([cell_start_ids_1,cell_start_ids_2])
            cell_start_ids = np.sort(cell_start_ids)
            cell_end_ids = np.where(np.array(tokens)==self.cell_e_id)[0]
            # print(cell_start_ids)
            # print(cell_end_ids)
            cur_batch_max_len = max(cur_batch_max_len, len(tokens))
            batch_tokens.append(tokens)
            batch_attention_mask.append([1] * len(tokens))
            batch_ranges.append( get_cell_ranges(
                                    row_start_ids,
                                    row_end_ids,
                                    cell_start_ids,
                                    cell_end_ids,
                                    bias) 
                                )
        if max_length is not None:
            max_length = min(cur_batch_max_len, max_length)
        else:
            max_length = cur_batch_max_len
        
        for i, tokens in enumerate(batch_tokens):
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                attention_mask = batch_attention_mask[i][:max_length]
            else:
                padding_length = (max_length - len(tokens))
                tokens = tokens + [self.pad_token_id] * padding_length
                attention_mask = batch_attention_mask[i] + [0] * padding_length
            batch_tokens[i] = tokens
            batch_attention_mask[i] = attention_mask
        
        batch_tokens_tensor = torch.tensor(batch_tokens, dtype=torch.int64).to(device)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.int64).to(device)

        return batch_tokens_tensor, batch_attention_mask, batch_ranges
        
        
    def call_decoder_cell_ranges_and_cell_span(self, tokens, bias=0):
        
        row_start_ids = np.where(np.array(tokens)==self.row_s_id)[0].tolist()
        row_end_ids = np.where(np.array(tokens)==self.row_e_id)[0].tolist()
        cell_start_ids_1 = np.where(np.array(tokens)==self.cell_s_id_1)[0].tolist()
        cell_start_ids_2 = np.where(np.array(tokens)==self.cell_s_id_2)[0].tolist()
        # print(cell_start_ids_1, cell_start_ids_2)
        cell_start_ids = np.concatenate([cell_start_ids_1,cell_start_ids_2])
        cell_start_ids = np.sort(cell_start_ids)
        cell_end_ids = np.where(np.array(tokens)==self.cell_e_id)[0].tolist()
        
        if len(row_start_ids)>len(row_end_ids):
            row_end_ids.append(len(tokens))
        if len(row_start_ids)!=len(row_end_ids):
            min_len = min(len(row_start_ids),len(row_end_ids))
            row_start_ids = row_start_ids[:min_len]
            row_end_ids = row_end_ids[:min_len]

        # assert len(row_start_ids)==len(row_end_ids), print("len(row_start_ids)==len(row_end_ids)", len(row_start_ids)==len(row_end_ids))

        if len(cell_start_ids)>len(cell_end_ids):
            cell_end_ids.append(len(tokens)-1)
        if len(cell_start_ids)!=len(cell_end_ids):
            min_len = min(len(cell_start_ids),len(cell_end_ids))
            cell_start_ids = cell_start_ids[:min_len]
            cell_end_ids = cell_end_ids[:min_len]
            # cell_end_ids.append(len(tokens)-1)
        
        assert len(cell_start_ids)==len(cell_end_ids), print("len(cell_start_ids)==len(cell_end_ids)", len(cell_start_ids)==len(cell_end_ids))
        cell_ranges,cell_spans, cell_texts = self.get_cell_ranges_and_span(
                                    row_start_ids,
                                    row_end_ids,
                                    cell_start_ids,
                                    cell_end_ids,
                                    tokens,
                                    bias) 
        if len(cell_ranges)==0:
            cell_ranges = [[[0+bias,len(tokens)-bias]]]
            cell_spans =[[[0,0]]]
            cell_texts=[[""]]
        return cell_ranges, cell_spans, cell_texts

    def get_cell_ranges_and_span(
            self,
            row_start_ids,
            row_end_ids,
            cell_start_ids,
            cell_end_ids,
            tokens,
            bias=0
        ):
        cur_c = 0
        ranges = list()
        spans = list()
        texts = list()
        for r_i,r_j in zip(row_start_ids,row_end_ids):
            range_list = list()
            span_list = list()
            text_list = list()
            for id,(c_i,c_j) in enumerate(zip(cell_start_ids[cur_c:], cell_end_ids[cur_c:])):
                if c_i>r_i and c_j<r_j:
                    range_list.append((int(c_i+bias), int(c_j+bias)))

                    token_temp = tokens[int(c_i):1+ int(c_j)]
                    
                    # text_list.append(self.detokenize(tokens[int(c_i+bias):1+ int(c_j+bias)]))
                    begin_id = 0
                    end_id = len(token_temp)-1
                    try:
                        end_id_e = token_temp.index(self.cell_e_id)
                        end_id = end_id_e if end_id_e!=-1 else end_id
                    except:
                        pass

                    r=1
                    c=1
                    text_temp = ""
                    if self.row_span_id in token_temp or self.col_span_id in token_temp: # 如果是span的话，那么起始id 变成 span_e 的位置
                        try:
                            begin_id_temp = token_temp.index(self.span_e)
                            begin_id = begin_id_temp if begin_id_temp!=-1 else begin_id
                        except:
                            pass
                        
                        match = self.detokenize(token_temp)
                        try:
                            if 'rowspan' in match:
                                r = int(re.findall(r'<iflytek_html_rowspan>(\d+)', match)[0])
                        except:
                            pass
                        try:
                            if 'colspan' in match:  
                                c = int(re.findall(r'<iflytek_html_colspan>(\d+)', match)[0])
                        except:
                            pass
                    span_list.append((r,c))
                    text_temp = self.detokenize(token_temp[begin_id+1:end_id])
                    if self.row_span_id in token_temp or self.col_span_id in token_temp:
                        print(text_temp, self.detokenize(token_temp))
                    text_list.append(text_temp)
                else:
                    cur_c=id+cur_c
                    break
                    
            ranges.append(range_list)
            spans.append(span_list)
            texts.append(text_list)
        return ranges,spans,texts



def get_cell_ranges(
        row_start_ids,
        row_end_ids,
        cell_start_ids,
        cell_end_ids,
        bias=0
    ):
    cur_c = 0
    ranges = list()
    for r_i,r_j in zip(row_start_ids,row_end_ids):
        range_list = list()
        for id,(c_i,c_j) in enumerate(zip(cell_start_ids[cur_c:], cell_end_ids[cur_c:])):
            if c_i>r_i and c_j<r_j:
                range_list.append((int(c_i+bias), int(c_j+bias)))
            else:
                cur_c=id+cur_c
                break
        ranges.append(range_list)
    
    return ranges

