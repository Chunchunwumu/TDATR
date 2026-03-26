import os
import random
import sys
sys.path.append("your project path")
import logging
import argparse
import json
import time
import traceback
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
import tqdm
import re
from generation_my.api2 import generate2
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
from TDATR_utils.utils import add_defaults
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
data_str = time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime())
logger = logging.getLogger(data_str+__name__)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from TDATR_utils.initialize import initialize_hulk, hydra_init
from TDATR_utils.utils import convert_namespace_to_omegaconf, omegaconf_no_object_check
from TDATR_utils.call_main import call_main

from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.global_context import global_context as gpc



@hydra.main(".", config_name="config")
def hydra_main(cfg) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg, **kwargs) -> float:
    # print(cfg.common)

    if HydraConfig.initialized():
        with open_dict(cfg):
            cfg.job_logging_cfg = OmegaConf.to_container(HydraConfig.get().job_logging, resolve=True)

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.set_struct(cfg, True)

    try:
        call_main(cfg, main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def encode_img(model, dataset, image_path, device):
    image, scale, img_shape,scale_r = dataset.load_image_padding_train(image_path)
    image_padding_shape = image.shape[1:]
    image = image.unsqueeze(0).to(device)
    image_embed, donut_outs = model.encode_img(image, image.clone().detach())
    donut_outs = [ i.half() for i in donut_outs]
    det_input = [donut_outs, scale, img_shape,scale_r, image_padding_shape]
    image_embed = image_embed.half()
    image_embed = image_embed.permute(1,0,2)    
    return image_embed, det_input
    
class Dataset_infer():
    def __init__(self):
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        
    def get_target_shape(self, dt_shape, target_shape_info):
        max_length = target_shape_info['max_length']
        patch_size = target_shape_info['patch_size']
        h, w = dt_shape[:2]
        
        #规则边长到固定倍数
        if h%patch_size!=0:
            h += (patch_size-h%patch_size)

        if w%patch_size!=0:
            w += (patch_size-w%patch_size)
        
        #规则边长到固定尺寸
        if h>w:
            nh = max_length
            nw = w/h*max_length
        else:
            nw = max_length
            nh = h/w*max_length
        
        target_shape = [int(nh), int(nw)]
        return target_shape
        
    def get_scale_ratio(self,info_path):
        data_set_list = {
            "SynthTabNet":2.5,
            "pubtabnet":4,
            "pubtables":2,
            "TabRecSet":2,
            'table_parsing_to_html':3,
        }
        
        r = 1
        for i,v in data_set_list.items():
            if i.lower() in info_path.lower():
                r = v
                break

        return r

    def recover_pred_cell_box2raw_image(self, image_shape, scale, scale_r, cell_boxes):
        # pdb.set_trace()
        
        h,w = image_shape
        h_scale, w_scale = scale
        cell_boxes = np.clip(cell_boxes, 0,1)
        cell_boxes = cell_boxes * np.array([[w,h,w,h]])/ np.array([[w_scale*scale_r,h_scale*scale_r,w_scale*scale_r,h_scale*scale_r]])

        cell_boxes = np.round(cell_boxes).astype(np.int32)
        return cell_boxes
    
    def load_image_padding_train(self,image_name):

        if isinstance(image_name, str):
            img = cv2.imread(image_name)
        else:
            img = image_name
        scale_r = self.get_scale_ratio(image_name)
        # scale_r = 1
        img = cv2.resize(img, dsize=None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC)
        
        ori_shape = img.shape[:2]
        min_patch = 256
        cur_shape = list(img.shape[:2])
        max_length = max(cur_shape)
        self_max_length = 2048
        #超出最大尺寸，缩放
        if max_length>=self_max_length:
            target_shape_info = {}
            target_shape_info['max_length'] = self_max_length
            target_shape_info['patch_size'] = 1
            cur_shape = self.get_target_shape(cur_shape, target_shape_info) 

        padding = [0, 0]
        if min(cur_shape) >= min_patch:
            cur_shape[0] = cur_shape[0] if cur_shape[0] % min_patch == 0 else cur_shape[0] + (min_patch - cur_shape[0] % min_patch)
            cur_shape[1] = cur_shape[1] if cur_shape[1] % min_patch == 0 else cur_shape[1] + (min_patch - cur_shape[1] % min_patch)
        elif max(cur_shape) >= min_patch:
            if cur_shape[0] > cur_shape[1]:
                cur_shape[0] = cur_shape[0] if cur_shape[0] % min_patch == 0 else cur_shape[0] + (min_patch - cur_shape[0] % min_patch)
                padding[1] = min_patch - cur_shape[1] 
            else:
                cur_shape[1] = cur_shape[1] if cur_shape[1] % min_patch == 0 else cur_shape[1] + (min_patch - cur_shape[1] % min_patch)
                padding[0] = min_patch - cur_shape[0]
        else:
            padding[0] = min_patch - cur_shape[0]
            padding[1] = min_patch - cur_shape[1]

        scale = np.array(cur_shape) / np.array(ori_shape)
        img = cv2.resize(img, (cur_shape[1], cur_shape[0]))
        img = cv2.copyMakeBorder(img, 0, padding[0], 0, padding[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_shape = img.shape[:2]
        img = self.to_tensor(img)
        return img, scale, img_shape, scale_r
    
    def process_cell_info(self, cell_boxes, cell_texts, cell_spans_html):
        c = 0
        cells_list = list()
        for id, cells in enumerate(cell_spans_html):
            cell_row_list = list()
            for c_id,cell in enumerate(cells):
                cell_temp = dict(
                    row_id = id,
                    text = cell_texts[id][c_id],
                    box = cell_boxes[c].tolist(),
                    span_html = cell,
                )
                c=c+1
                cell_row_list.append(cell_temp)
            cells_list.append(cell_row_list)
        return cells_list

def main(cfg) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    add_defaults(cfg)
    if cfg.common.npu:
        from TDATR_utils.npu import set_npu
        set_npu()
    initialize_hulk(cfg)
    if cfg.distributed_training.distributed_rank == 0 and "job_logging_cfg" in cfg:
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    # Check cfg
    cfg.model_parallel.recompute_granularity = None
    cfg.model_parallel.sequence_parallel = False
    cfg.model.parallel_output = False

    if gpc.get_world_size(ParallelMode.DATA) > 1:
        raise RuntimeError("The generation function does not support data parallelism!")
    
    from TDATR.models.mini_gpt4_ipt_v2 import MiniGPT4
    from TDATR.models.detect.structures_.instance_data import InstanceData
    from TDATR.models.detect.structures_.det_data_sample import DetDataSample
    dataset = Dataset_infer()    
    
    eos_token = '<end>'
    device = torch.device("cuda")

    model = MiniGPT4(cfg).half()
    tokenizer = model.ipt_tokenizer
    logger.info("model: {}".format(model.__class__.__name__))

    model.eval()
    model = model.to(device=device)
    
    # must be train state
    model.cfgi_decoder.neck.train()
    model.cfgi_decoder.encoder.train()
    with open(cfg.generation.prompt_path, 'r') as f:
        samples = json.load(f)

    output_base_dir = "output"
    output_dir = os.path.join(output_base_dir, "infer_TDATR")
    os.makedirs(output_dir, exist_ok=True)

    out_file_name = os.path.splitext(os.path.split(cfg.generation.prompt_path)[1])[0]
    output_vis_dir = os.path.join( output_dir, out_file_name, "out_vis" )
    output_dir = os.path.join( output_dir, out_file_name, "out_jsons" )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    for image_path in tqdm.tqdm(samples):
        logger.info(f"image_path: {image_path}")
        save_path = os.path.join(output_dir,os.path.basename(image_path)+".json")
        
        if os.path.exists(save_path):
            continue

        image_embed, det_input = encode_img(model, dataset, image_path, device)
        _, scale, img_shape, raw_scale,image_padding_shape = det_input
        
        data_sample = DetDataSample()
        instance_data = InstanceData()
        data_sample.gt_instances = instance_data

        img_meta = {}
        img_meta['img_shape'] = img_shape # h,w
        img_meta['batch_input_shape'] = img_shape
        data_sample.set_metainfo(img_meta)
        gt_inst = data_sample
        query_text = "将图片中的表格转换为HTML语言。<iflytek_ret>"
        raw_query, raw_answer, clear_answer, gen_tokens, embs, context_length_ywx, \
            cell_boxes_pred, cell_span_html, cell_texts = single_prompt_process_cfgi(model,tokenizer,eos_token, 
                                        query_text, image_embed,
                                        max_new_tokens=gpc.config.generation.max_len, 
                                        sampling_topk=gpc.config.generation.sampling_topk, 
                                        sampling_topp=gpc.config.generation.sampling_topp, 
                                        temperature=gpc.config.generation.temperature, 
                                        max_length=gpc.config.generation.max_len, 
                                        random_seed=gpc.config.task.seed,
                                        image_shape=image_padding_shape,
                                        donuts_out=det_input[0],
                                        gt_inst=gt_inst,
                                        )
        cell_boxes_pred = dataset.recover_pred_cell_box2raw_image( img_shape, scale, raw_scale, cell_boxes_pred )
        image = cv2.imread(image_path)
        for cell in cell_boxes_pred:
            image = cv2.rectangle(image, cell[:2].tolist(), cell[2:].tolist(), (0,0,255), 3)
        cv2.imwrite(os.path.join(output_vis_dir, os.path.basename(image_path)), image)
        cell_list = dataset.process_cell_info(cell_boxes_pred, cell_texts, cell_span_html)
        answer= dict(query=query_text,
                    clear_answer=clear_answer,
                    raw_answer=raw_answer,
                    cells = cell_list,)
        logger.info(json.dumps(dict(query=query_text,
                                clear_answer=clear_answer
                            ),
                            ensure_ascii=False))
        
        ans_info = dict(
            image_path = image_path,
            answer=answer,
        )
        
        
        with open(save_path, "w") as f:
            f.write(json.dumps(ans_info, indent=4 ,ensure_ascii=False))
        logger.info(f"save result: {save_path}")
        logger.info(f"save vis in: {os.path.join(output_vis_dir, os.path.basename(image_path))}")



@torch.no_grad()
def get_context_emb(model, prompt, image_tensor):
    img_embeds = image_tensor

    seg_embeds = list()
    seg_tokens = list()
    seg_embed, seg_token, *_ = model.encode_text([prompt])
    seg_embeds.append(seg_embed)
    seg_tokens.append(seg_token)

    mixed_embeds = torch.cat(seg_embeds, dim=1)
    mixed_tokens = torch.cat(seg_tokens, dim=1)

    return prompt, mixed_embeds, mixed_tokens, img_embeds #[BLC]

@torch.no_grad()
def pad_tokens(model, tokenizer, tokens, inputs_embeds, inputs_embeds_length, tokens_to_generate):
    max_prompt_len = torch.max(inputs_embeds_length).item()
    _B, _L = tokens.shape

    pad_tokens = torch.ones((_B, tokens_to_generate), device = tokens.device, dtype=tokens.dtype) * tokenizer.pad_id #[B, L_pad]
    pad_embeddings = model.ipt_model.embedding(pad_tokens, None).permute(1, 0, 2) #[B, L_pad, emd_dim]

    tokens = torch.cat([tokens, pad_tokens], dim = 1)
    inputs_embeds = torch.cat([inputs_embeds, pad_embeddings], dim=1)
    return tokens, inputs_embeds, inputs_embeds_length


def single_prompt_process_cfgi(model,tokenizer, eos_token, prompt, image_tensor, max_new_tokens=1024, sampling_topk=4, sampling_topp=0.0, 
               temperature=0.5, penalty=1.7, max_length=8192, random_seed=42, image_shape=None,donuts_out=None,gt_inst=None):

    prompt, embs, tokens, img_embeds = get_context_emb(model, prompt, image_tensor)  # [B, L, emb_dim]
    current_len = embs.shape[1] #B L C
    if current_len - max_length > 0:
        print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
    begin_idx = max(0, current_len - max_length)

    embs = embs[:, begin_idx:]
    tokens = tokens[:, begin_idx:]
    _B, _L, _D = embs.shape
    embs_length = torch.ones((_B,), device=embs.device, dtype=torch.long) * _L  # [B, L]
    


    bs_size = embs.shape[0]
    assert bs_size == 1, 'parallel decode is not implemented'

    with torch.no_grad():
        
        tokens, inputs_embeds, inputs_embeds_length = pad_tokens(model, tokenizer, tokens, embs, embs_length, max_new_tokens)
        (outputs,hidden_state_list),context_length_ywx  = generate2(
            model=model,
            tokenizer=tokenizer,
            tokens = tokens,
            inputs_embeds=inputs_embeds,
            img_embeds=img_embeds,
            inputs_embeds_length=inputs_embeds_length,
            tokens_to_generate=max_new_tokens,
            return_output_log_probs=False,
            top_k_sampling=sampling_topk,
            top_p_sampling=sampling_topp,
            temperature=temperature,
            penalty=penalty, 
            add_BOS=False,
            random_seed=random_seed
        )
        hidden_state_list = torch.concat(hidden_state_list, dim=1) 
        hidden_state_list = torch.split(hidden_state_list, 1, dim=-1)
        hidden_state_list = [i[...,0] for i in hidden_state_list]
        
        # 需要在此基础上，返回隐藏层状态
        raw_answer = outputs[0]["generate"]
        gen_tokens = outputs[0]["gen_token"]

        out_answer = raw_answer.replace(eos_token, "")
        out_answer = out_answer.replace('<iflytek_ret>', '\n')

        cell_ranges, cell_spans,cell_texts = tokenizer.call_decoder_cell_ranges_and_cell_span(gen_tokens, bias=(current_len-1))
        cfgi_hidden_state,row_position,pred_row,pred_col = model.aggregate_cell_tokens( hidden_state_list, cell_range_ids=cell_ranges)
        
        
        row_mask1 = torch.where(pred_row.sigmoid()>0.5,1,0).to(pred_row)
        if torch.sum(row_mask1)==0:
            row_mask1 = torch.eye(row_mask1.shape[0])
        col_mask1 = torch.where(pred_col.sigmoid()>0.5,1,0).to(pred_col)
        if torch.sum(col_mask1)==0:
            col_mask1 = torch.eye(col_mask1.shape[0])

        cell_hidden_states,_ = model.cfgi_ipt_model.transformer(
                        hidden_states=cfgi_hidden_state,   
                        position_ids=None,
                        attention_mask=None,
                        kv_hidden_states=img_embeds,
                        row_col_positions=row_position,
                        row_same_mask=row_mask1[...,None],
                        col_same_mask=col_mask1[...,None],
                    )
        cell_boxes_pred = \
                model.cfgi_decoder.forward(cell_hidden_states, img_embeds.transpose(0,1), 
                                            image_shape, pred_row,pred_col,
                                            row_position,donuts_out, gt_inst)
        
        cell_boxes_pred = bbox_cxcywh_to_xyxy(cell_boxes_pred[-1])

        cell_boxes_pred = cell_boxes_pred.detach().cpu().numpy()[0]
    try:
        out_answer = reverseFormat(out_answer)
    except:
        pass
    return prompt, raw_answer, out_answer, gen_tokens, embs, context_length_ywx, \
        cell_boxes_pred, cell_spans, cell_texts

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

def reverseFormat(content):
    content = content.replace('<iflytek_html_html_s>', '<html>')
    content = content.replace('<iflytek_html_html_e>', '</html>')
    content = content.replace('<iflytek_html_body_s>', '<body>')
    content = content.replace('<iflytek_html_body_e>', '</body>')
    content = content.replace('<iflytek_html_table_s>', '<table>')
    content = content.replace('<iflytek_html_table_e>', '</table>')
    content = content.replace('<iflytek_html_thead_s>', '<thead>')
    content = content.replace('<iflytek_html_thead_e>', '</thead>')
    content = content.replace('<iflytek_html_tbody_s>', '<tbody>')
    content = content.replace('<iflytek_html_tbody_e>', '</tbody>')
    content = content.replace('<iflytek_html_td_s>', '<td>')
    content = content.replace('<iflytek_html_td_e>', '</td>')
    content = content.replace('<iflytek_html_tr_s>', '<tr>')
    content = content.replace('<iflytek_html_tr_e>', '</tr>')
    content = content.replace('<iflytek_br>', '<br>')
    pattern = re.compile(r'(<iflytek_html_span_s>(.*?)<iflytek_html_span_e>)')
    matched = pattern.findall(content)
    for match in matched:
        match = match[0]
        try:
            if 'rowspan' in match and 'colspan' in match:
                col_span_num = re.findall(r'<iflytek_html_colspan>(\d+)', match)[0]
                row_span_num = re.findall(r'<iflytek_html_rowspan>(\d+)', match)[0]
                if match.index('rowspan') > match.index('colspan'):
                    new_str = match.replace(f'<iflytek_html_span_s><iflytek_html_colspan>{col_span_num}', f'<td colspan={col_span_num} ')
                    new_str = new_str.replace(f'<iflytek_html_rowspan>{row_span_num}<iflytek_html_span_e>', f'rowspan={row_span_num}>')
                else:
                    new_str = match.replace(f'<iflytek_html_colspan>{col_span_num}<iflytek_html_span_e>', f'colspan={col_span_num}>')
                    new_str = new_str.replace(f'<iflytek_html_span_s><iflytek_html_rowspan>{row_span_num}', f'<td rowspan={row_span_num} ')
            elif 'rowspan' in match:
                row_span_num = re.findall(r'<iflytek_html_rowspan>(\d+)', match)[0]
                new_str = match.replace(f'<iflytek_html_span_s><iflytek_html_rowspan>{row_span_num}<iflytek_html_span_e>', f'<td rowspan={row_span_num}>')
            elif 'colspan' in match:
                
                col_span_num = re.findall(r'<iflytek_html_colspan>(\d+)', match)[0]
                new_str = match.replace(f'<iflytek_html_span_s><iflytek_html_colspan>{col_span_num}<iflytek_html_span_e>', f'<td colspan={col_span_num}>')
        except:
            traceback.print_exc()
            pass
        content = content.replace(match, new_str)

    content = content.replace('<iflytek_line_equation_s>', '')
    content = content.replace('<iflytek_line_equation_e>', '')
    content = content.replace('<iflytek_inline_equation_s>', '')
    content = content.replace('<iflytek_inline_equation_e>', '')
    content = content.replace('<iflytek_unk>', '')
    content = content.replace('<iflytek_left_brace>', '{')
    content = content.replace('<iflytek_right_brace>', '}')

    return content



def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
