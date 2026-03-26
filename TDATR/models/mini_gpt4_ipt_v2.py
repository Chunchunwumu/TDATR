import os
import logging
import random
import numpy as np
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import copy
from TDATR.tokenizers.bbox_tokenizer import BboxTokenizer
from TDATR_utils.global_context import global_context as gpc

from TDATR_utils.dataclass import HulkDataclass
from TDATR.models.detect.channel_mapper import ChannelMapper
from TDATR.models.detect.hybrid_encoder import HybridEncoder

from TDATR.models.detect.dino_layers import SinePositionalEncoding, DeformableDetrTransformerDecoderLayer

from TDATR.models.blip2 import Blip2Base
from TDATR.models.ipt_model import parallel_lm_logits
from TDATR.models.ipt_v4 import IPTAttConfigV4, IPTV4Model
from TDATR.models.ipt_v4_cfgi import  IPTV4Model as IPTV4Model_cfgi

from TDATR.models.pos_utils import get_2d_sincos_pos_embed,get_abs_pos_rectangle,get_1d_sincos_pos_embed_from_grid
from TDATR_utils.global_variables import ParallelMode

logger = logging.getLogger(__name__)
import collections


from TDATR.models.swin_transformer_tp import create_tp_swin_transformer
from TDATR_utils.models import MixedFusedLayerNorm as LayerNorm
import matplotlib.pyplot as plt

@dataclass
class MiniGPT4LoRAConfig(HulkDataclass):
    apply_lora: bool = field(
        default=False,
        metadata={
            "help": "fine-tune the model with LoRA if enabled."
        }
    )
    only_train_lora: bool = field(
        default=False, 
        metadata={"help": "only lora is trained, vit&qformer will be fixed"}
    )
    lora_rank: int = field(
        default=0, metadata={"help": "lora hidden layer dimension"}
    )
    lora_alpha: int = field(
        default=128, metadata={"help": "lora attn alpha"}
    )
    lora_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for lora layers"}
    )
    adapt_q: bool = field(
        default=False, metadata={"help": "adapting the attention query weights"}
    )
    adapt_k: bool = field(
        default=False, metadata={"help": "adapting the attention key weights"}
    )
    adapt_v: bool = field(
        default=False, metadata={"help": "adapting the attention value weights"}
    )
    adapt_o: bool = field(
        default=False, metadata={"help": "adapting the attention output project weights"}
    )
    adapt_fc1: bool = field(
        default=False, metadata={"help": "adapting the first linear in FFN module"}
    )
    adapt_fc2: bool = field(
        default=False, metadata={"help": "adapting the first second in FFN module"}
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "merge two branches during evaluation if enabled"}
    )
    from_pretrained: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load pretrained LoRA model if from_pretrained is not None"
        }
    )

@dataclass
class CFGIConfig(IPTAttConfigV4):
    scaled_upper_triang_masked_softmax_fusion:Optional[bool]=field(default=False)
    apply_residual_connection_post_layernorm:Optional[bool]=field(default=False)
    end_sym:Optional[str]=field(
        default="<iflytek_end>"
    ) 
    cell_points:Optional[int]=field(
        default=4
    )
    logical_dim:Optional[int]=field(
        default=2
    )
    logical_position_type:Optional[str]=field(
        default="rope" # "learning"
    )
    cfgi_rope:Optional[bool]= field(
        default=True,
    )

@dataclass
class MiniGPT4Config(IPTAttConfigV4):
    vit_model: Optional[str] = field(
        default="eva_clip_g",
        metadata={"help": "vit model name"}
    )
    use_vit_encoder: Optional[bool]= field(
        default=True,
        metadata={"help": "whether use vit encoder"}
    )
    donut_model: Optional[str] = field(
        # default="/train34/mmu/permanent/wxyu2/pretrain/haowu16/donut-internal-ChineseBart-460w-pretrain-NewIO-pretrain",
        default="/train34/mmu/permanent/wxyu2/proj/kosmos/kosmos_hw/donut-internal-ChineseBart-460w-pretrain-NewIO-pretrain",
        metadata={"help": "donut model path"}
    )
    use_donut_encoder: Optional[bool]= field(
        default=False,
        metadata={"help": "whether use donut encoder"}
    )
    use_dino: Optional[bool]= field(
        default=False,
        metadata={"help": "whether use dino detector"}
    )
    use_dino_mask: Optional[bool]= field(
        default=False,
        metadata={"help": "whether use dino mask detector"}
    )
    use_dab: Optional[bool]= field(
        default=False,
        metadata={"help": "whether use dab detector"}
    )
    freeze_ipt_proj: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the parameters of donut"}
    )
    freeze_ipt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the parameters of donut"}
    )
    freeze_donut_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the parameters of donut encoder"}
    )
    freeze_donut_proj: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the parameters of donut proj"}
    )
    freeze_backbone: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone"}
    )
    freeze_bbox_embed: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone"}
    )
    freeze_ocr_embed: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone"}
    )
    qformer_name: Optional[str] = field(
        default="bert-base-uncased", 
        metadata={"help": "qformer type"}
    )
    q_former_model: Optional[str] = field(
        default="",
        metadata={"help": "the path of QFormer"}
    )
    ckpt: Optional[str] = field(
        default="",
        metadata={"help": "weights of MiniGPT-4"}
    )
    freeze_vit: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the parameters of ViT"}
    )
    freeze_qformer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the parameters of QFormer"}
    )
    vit_precision: Optional[str] = field(
        default="fp16",
        metadata={"help": "num_query_token"}
    )
    image_size: Optional[int] = field(
        default=224,
        metadata={"help": "image size"}
    )
    num_query_token: Optional[int] = field(
        default=32,
        metadata={"help": "num_query_token"}
    )
    drop_path_rate: Optional[float] = field(
        default=0,
        metadata={"help": "drop_path_rate"}
    )
    bos_token: Optional[str] = field(
        default="<iflytek_end>", 
    )
    eos_token: Optional[str] = field(
        default="<iflytek_end>", 
    )
    pad_token: Optional[str] = field(
        default="<iflytek_pad>", 
    )
    ibos_token: Optional[str] = field(
        default="<Img>", 
    )
    ieos_token: Optional[str] = field(
        default="</Img>", 
    )
    add_pos_embed: Optional[bool] = field(
        default=False
    )
    lora: MiniGPT4LoRAConfig = field(
        default=MiniGPT4LoRAConfig, 
        metadata={"help": "lora config for minigpt4"}
    )
    use_cfgi: Optional[bool]= field(
        default=True,
        metadata={"help": "whether use cell fine-grained information decoder"}
    )
    CFGI_CFG: CFGIConfig = field(
        default=CFGIConfig
    )


class MiniGPT4(Blip2Base):
    def __init__(self, cfg: MiniGPT4Config):
        super().__init__()
        self.cfg = cfg
        self.set_trick_cfgs()    # set config for tricks

        self.init_donut()
        self.init_ipt()
        self.init_bbox_token_embed()   
        self.init_cfgi_decoder()

        #FFN: merge feat_32 & feat_16
        self.conv_feat32 = nn.Conv2d(2048, 2048, 1, 1, 0)
        self.conv_feat16 = nn.Conv2d(1024, 2048, 3, 1, 1)
        self.conv_proj =  nn.Conv2d(2048, 2048, 3, 1, 1)

        self.pos_embed_kv_16 = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(2048, 160)).float()
        ).requires_grad_(False) 


        grid = np.arange(4096, dtype=np.float32)

        self.pos_embed_text = nn.Parameter(
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(cfg.model.embed_dim, grid)).float()
        ).requires_grad_(False) 



        if cfg.model.ckpt:
            self.load_ckpt()
            
        self.pad_token = self.ipt_tokenizer.pad_id
        
        self._count_image = 0
        self.forwad_bsz_trans = lambda x, device: torch.stack(x,0).squeeze(1).to(device)

    
    def set_trick_cfgs(self, ):
        cfg = self.cfg.model
        self.ibos_token = cfg.ibos_token
        self.ieos_token = cfg.ieos_token
        
        
        self.qformer_pos_embeds = None

    def init_donut(self, ):

        self.donut_model = create_tp_swin_transformer(
                img_size=2048, patch_size=4, embed_dim=256, depths=(2, 2, 18, 2), 
                num_heads=(4, 8, 16, 32), window_size=8)
    
        self.donut_model = self.donut_model.to(torch.cuda.current_device())

    def init_ipt(self, ):
        cfg = self.cfg
        logger.info('Loading IPT')

        self.ipt_tokenizer = BboxTokenizer(self.cfg.tokenizer)

        self.ipt_model = IPTV4Model.build_model(cfg.model)
        logger.info('Loading IPT Done')


    def init_cfgi_decoder(self,):
        if self.cfg.model.use_cfgi:
            logger.info('Loading cell fine-grained information IPT')
            self.cfgi_ipt_model = IPTV4Model_cfgi.build_model(self.cfg.model.CFGI_CFG)
            logger.info('Loading cell fine-grained information IPT Done')
            self.cfgi_decoder = CellFineGrainedInfoParsing( self.cfg.model.CFGI_CFG.cell_points,
                                                        self.cfg.model.CFGI_CFG.logical_dim,
                                                        self.cfg.model.CFGI_CFG.embed_dim )    
            
        
            self.cfgi_fuse = nn.Linear(6,1)
            self.cfgi_ln = nn.LayerNorm(self.cfg.model.CFGI_CFG.embed_dim)



    def init_bbox_token_embed(self):
        logger.info("Loading Bbox Embedding Layer")
        self.ipt_bbox_embedding = IPTBboxEmbedding(self.ipt_tokenizer.added_vocab_size, self.cfg.model.embed_dim, self.ipt_model.embedding)      

    def load_ckpt(self, ):
        cfg = self.cfg.model
        logger.info("Load Checkpoint: {}".format(cfg.ckpt))

        missing_keys, unexpected_keys = None, None
        state_dict = torch.load(cfg.ckpt, map_location="cpu")
        
        msg = self.load_state_dict(state_dict=state_dict, strict=False)
        missing_keys = set(msg.missing_keys) if missing_keys is None else (missing_keys & set(msg.missing_keys))
        unexpected_keys = set(msg.unexpected_keys) if unexpected_keys is None else (unexpected_keys |  set(msg.unexpected_keys))

        missing_keys = {item for item in missing_keys if 'relative_position_index' not in item}
            
        # NOTE: Only use rank 0 to load weights into cpu memory
        # if gpc.get_global_rank() == 0:
            # state_dict = torch.load(cfg.ckpt, map_location="cpu")
        #     # data = np.zeros((111, 2048), np.float32) # 多的是表格结构token
        #     # key = 'ipt_bbox_embedding.bbox_embedding.weight'
        #     # for k in ['ipt_bbox_embedding.bbox_embedding.weight', 'ipt_model.embedding.word_embeddings.weight']:
        #     #     arr = state_dict[k].numpy().copy()
        #     #     data += arr.mean()
        #     # state_dict[key] = torch.cat([state_dict[key], torch.from_numpy(data)], 0)
            

        #     # state_dict_copy = copy.deepcopy(state_dict)
        #     # for k in state_dict_copy.keys():
        #     #     if 'cross_attns' in k or 'pos_embed_kv_16' in k:
        #     #         del state_dict[k]
        #     # del state_dict_copy
            
        #     '''
        #     data = np.zeros((65, 2048), np.float32)
        #     key = 'ipt_bbox_embedding.bbox_embedding.weight'
        #     for k in ['ipt_bbox_embedding.bbox_embedding.weight', 'ipt_model.embedding.word_embeddings.weight']:
        #         arr = state_dict[k].numpy().copy()
        #         data += arr.mean()
        #     state_dict[key] = torch.cat([state_dict[key], torch.from_numpy(data)], 0)
        #     '''

        #     if len(state_dict) >= 100:
        #         num_parts = torch.tensor([8]).cuda()
        #     else:
        #         num_parts = torch.tensor([1]).cuda()
        #     num_per_part = int(math.ceil(len(state_dict) * 1.0 / num_parts.item()))
        #     state_keys = list(state_dict.keys())

        # num_parts = distributed_utils.broadcast_object(
        #             obj=num_parts,
        #             src_rank=gpc.get_ranks_in_group(ParallelMode.GLOBAL)[0], 
        #             group=gpc.get_group(ParallelMode.GLOBAL),
        #             dist_device=torch.cuda.current_device(),
        #             cur_rank=gpc.get_global_rank(),
        #         )
        # num_parts = num_parts.item()
        
        # logger.info(f'seperate state_dict to {num_parts} parts')
        
        # for part_idx in range(num_parts):
        #     if gpc.get_global_rank() == 0:
        #         logger.info(f'load state_dict part {part_idx+1}/{num_parts}')
        #         _beg = part_idx * num_per_part
        #         _end = min((part_idx + 1) * num_per_part, len(state_dict))
        #         ks = [state_keys[i] for i in range(_beg, _end)]
        #         state_dict_part = {k: state_dict[k] for k in ks}

        #     # flag = False
        #     # for k in state_dict_part:
        #     #     if "cfgi_fuse" in k:
        #     #         print("***********************************************")
        #     #         print(state_dict_part[k])
        #     #         flag = True
        #     #         break
        #     # if flag:
        #     #     break  

        #     # NOTE: Distribute the weight to ranks other than rank 0. 
        #     # After broadcast, the GPU memory occupation of rank 0 will be lower than that of other ranks, 
        #     # because its corresponding weight has not been loaded into the corresponding GPU memory;
        #     state_dict_part = distributed_utils.broadcast_object(
        #         obj=state_dict_part,
        #         src_rank=gpc.get_ranks_in_group(ParallelMode.GLOBAL)[0], 
        #         group=gpc.get_group(ParallelMode.GLOBAL),
        #         dist_device=torch.cuda.current_device(),
        #         cur_rank=gpc.get_global_rank(),
        #     )


            # msg = self.load_state_dict(state_dict=state_dict_part, strict=False)
            # missing_keys = set(msg.missing_keys) if missing_keys is None else (missing_keys & set(msg.missing_keys))
            # unexpected_keys = set(msg.unexpected_keys) if unexpected_keys is None else (unexpected_keys |  set(msg.unexpected_keys))

            # missing_keys = {item for item in missing_keys if 'relative_position_index' not in item}
            
            # # NOTE: Release GPU memory occupied by pre-training weights
            # state_dict_part = None
            # torch.cuda.empty_cache()
        
        if gpc.get_global_rank() == 0:
            logger.info(f'missing_keys:\n {missing_keys}\n')
            logger.info(f'unexpected_keys:\n {unexpected_keys}\n')

        logger.info('------miss keys----')
        for key in missing_keys:
            if 'cross_attns' not in key:
                logger.info(key)

        logger.info('------unexpected_keys----')
        for key in unexpected_keys:
            if 'ipt_model.transformer.layers' not in key:
                logger.info(key)

        
        if 'ipt_bbox_embedding.bbox_embedding.weight' in missing_keys:
            mean_ipt_embed = torch.mean(self.ipt_model.embedding.word_embeddings.weight.data, dim=0, keepdim=True)  # 1, 2560
            self.ipt_bbox_embedding.bbox_embedding.weight.data[:] = mean_ipt_embed
            logger.info('init bbox embedding with ipt embedding mean')

    
        


    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    def encode_img(self, images, donut_images=None):
        cfg = self.cfg
        B = images.size(0)
        donut_outs = None
        with self.maybe_autocast():
            input_sources = []
            final_shape = []
            if self.cfg.model.use_donut_encoder and donut_images is not None and self.donut_model is not None:
                donut_outs = self.donut_model(donut_images)

                feat_16 = self.conv_proj(self._upsample_add(self.conv_feat32(donut_outs[3]), self.conv_feat16(donut_outs[2])))

                feat_shape_16 = feat_16.shape[2:]
                donut_image_embeds_16 = feat_16.view(B, 2048, -1).permute(0, 2, 1)

            pos_embed_kv_16 = get_abs_pos_rectangle(self.pos_embed_kv_16, feat_shape_16) # [L, emb_dim] 
            encode_kv_16 = donut_image_embeds_16 + pos_embed_kv_16

        return encode_kv_16, donut_outs

    def encode_text(self, texts):
        text_tokens, attn_masks = self.ipt_tokenizer(texts)
        text_tokens = text_tokens.to(self.device)
        attn_masks = attn_masks.to(self.device)
        text_embeds = self.ipt_bbox_embedding(text_tokens, None)
        return text_embeds, text_tokens, attn_masks
      
    def aggregate_cell_tokens(self, hidden_state, cell_range_ids=None):
        # TODO_cxqin 是否需要融合浅层的特征，通过可学习权重来调节融合权重

        """
        aggregate cell tokens based on cell token range

        parameters:
            hidden_state: HTML hidden state from HTML Decoder (B,N,dim)
            cell_range_ids: Cell tokens subscript range

        return:
            cell_tokens: the aggregatation result of cell tokens(B,N1,dim)
        """
        # TODO-CXQIN cell 是否用一个单一的token足以表示
        # print([i.shape for i in hidden_state])
        hidden_state = self.cfgi_fuse(torch.stack(hidden_state, dim=-1))[...,0] # 融合各个layer的特征
        hidden_state = self.cfgi_ln(hidden_state)
        all_row_ids = [] #里面每一个元素是单元格格的简化span
        cell_rep_tokens = []
        cell_range_ids = [cell_range_ids]
        for hs,cell_ranges in zip(hidden_state, cell_range_ids):
            row_ids = []
            cell_mean_tokens = []
            for row_id,cells in enumerate(cell_ranges):
                for cell in cells:
                    cell_mean_tokens.append( hs[cell[1]] )
            row_ids = torch.tensor(row_ids).to(hidden_state)
            cell_rep_tokens.append(torch.stack(cell_mean_tokens, dim=0))
            all_row_ids.append(row_ids)


        # padding 到同样的长度
        max_len = max([i.shape[0] for i in all_row_ids])

        pad_all_row_ids = []
        pad_cell_rep_tokens = []
        for row_ids, cell_tokens in zip(all_row_ids, cell_rep_tokens):
            if max_len!=row_ids.shape[0]:
                cell_tokens = torch.concat( [cell_tokens, torch.zeros(((max_len-row_ids.shape[0]), cell_tokens.shape[1])).to(cell_tokens)],dim=0 )
                row_ids = torch.concat([row_ids, torch.zeros(((max_len-row_ids.shape[0]), row_ids.shape[1])).to(row_ids)])
            pad_all_row_ids.append(row_ids)
            pad_cell_rep_tokens.append(cell_tokens)

        row_position = torch.stack(pad_all_row_ids, dim=1) #TODO 加入row_position编码

        cell_tokens = torch.stack(pad_cell_rep_tokens, dim=1) 
        

        
        pred_row = self.cfgi_decoder.compute_adjacency_matrix(self.cfgi_decoder.row_head[-1](cell_tokens.transpose(0,1)))[0]
        pred_col = self.cfgi_decoder.compute_adjacency_matrix(self.cfgi_decoder.col_head[-1](cell_tokens.transpose(0,1)))[0]

        return cell_tokens, row_position,pred_row,pred_col
    
    def forward(self, images=None, donut_images=None, image_indices=None, input_tokens=None, targets=None, loss_masks=None, 
                    position_ids=None, attention_masks=None, textlines=None):
        cfg = self.cfg
        
        textlines["train_inst"].gt_instances.bboxes = textlines["train_inst"].gt_instances.bboxes.to(self.device)
        textlines["train_inst"].gt_instances.labels = textlines["train_inst"].gt_instances.labels.to(self.device)


        if isinstance(images, list):
            images = self.forwad_bsz_trans(images,self.device)
            donut_images = self.forwad_bsz_trans(donut_images,self.device)
            input_tokens = self.forwad_bsz_trans(input_tokens,self.device)
            targets = self.forwad_bsz_trans(targets,self.device)
            loss_masks = self.forwad_bsz_trans(loss_masks,self.device)
        else:
            images = images.to(self.device) #[2, 3, 448, 448]
            input_tokens = input_tokens.to(self.device) #[1, 2048]
            donut_images = donut_images.to(self.device)
            targets = targets.to(self.device) #[1, 2048]
            loss_masks = loss_masks.to(self.device) #[1, 2048]
            
        if isinstance(position_ids,list): # if the position is tensor 
            assert position_ids[0] is None, "forward_batch > 1 does not support different tasks combination !!!!"
            position_ids, attention_masks =  None, None
        else:
            position_ids = position_ids.to(self.device) if position_ids is not None else None
            attention_masks = attention_masks.to(self.device) if attention_masks is not None else None
                

        img_embeds_16, donut_outs = self.encode_img(images, donut_images) #[2, 262, 2560] 
        img_embeds_16 = img_embeds_16.permute(1,0,2)


        result_infos = dict()

        txt_embeds = self.ipt_bbox_embedding(input_tokens, position_ids) #input_tokens 前262是0，后面是非0(代表text)
        txt_embeds = txt_embeds.permute(1, 0, 2) #[2048, 1, 2560]

        input_embeds = txt_embeds 

        
        if gpc.config.common.bf16:
            input_embeds = input_embeds.bfloat16()
            img_embeds_16 = img_embeds_16.bfloat16()
            # img_embeds_32 = img_embeds_32.bfloat16()

        elif gpc.config.common.fp16:
            input_embeds = input_embeds.half()
            img_embeds_16 = img_embeds_16.half()
            # img_embeds_32 = img_embeds_32.half()
        else:
            input_embeds = input_embeds.half()
            img_embeds_16 = img_embeds_16.half()
            # img_embeds_32 = img_embeds_32.half()

        
        hidden_states, hs_list = self.ipt_model.transformer( 
            hidden_states=input_embeds,   
            position_ids=position_ids,
            attention_mask=attention_masks,
            kv_hidden_states=img_embeds_16,
        )

        logits_parallel = self.ipt_bbox_embedding.get_logits_parallel(hidden_states)
        result_infos.update({'logits_parallel': logits_parallel, 'targets': targets, "loss_mask": loss_masks})

        if cfg.use_cfgi:
            cell_aggregate_states,row_position,pred_row,pred_col = self.aggregate_cell_tokens( hs_list, cell_range_ids=textlines["cell_ranges"], cell_spans= textlines["cell_rc_spans"] ) # TODO-cxqin cell range ids标签生成
            
            row_mask1 = torch.where(pred_row.sigmoid()>0.5,1,0).to(pred_row)
            if torch.sum(row_mask1)==0:
                row_mask1 = torch.eye(row_mask1.shape[0])
            col_mask1 = torch.where(pred_col.sigmoid()>0.5,1,0).to(pred_col)
            if torch.sum(col_mask1)==0:
                col_mask1 = torch.eye(col_mask1.shape[0])
            cell_hidden_states,_ = self.cfgi_ipt_model.transformer(
                    hidden_states=cell_aggregate_states,   
                    position_ids=position_ids,
                    attention_mask=attention_masks,
                    kv_hidden_states=img_embeds_16,
                    row_col_positions=row_position,
                    row_same_mask=row_mask1[...,None],
                    col_same_mask=col_mask1[...,None],
                )
            cell_boxes_pred = self.cfgi_decoder.forward(cell_hidden_states,img_embeds_16.transpose(0,1),images.shape[2:], 
                                                                                                                  pred_row,pred_col,
                                                                                                                  row_position,donut_outs, textlines["train_inst"])
        return result_infos
    @classmethod
    def build_model(cls, cfg: MiniGPT4Config, task):
        model = cls(cfg)
        return model


class CellFineGrainedInfoParsing(nn.Module):
    def __init__(self, cell_box_point_nums, logical_dim, embed_dim) -> None:
        super().__init__()
        logger.info('Loading cell fine-grained information decoder ouput')
        
        self.cell_att_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim),
        )
        self.image_att_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim),
        )
        self.cell_token_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 256),
        )
        
        col_head = list()
        row_head = list()

        for _ in range(1):
            cell_col = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(embed_dim//2, embed_dim//2),
                    nn.LayerNorm(embed_dim//2)
                )
            cell_row = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(embed_dim//2, embed_dim//2),
                    nn.LayerNorm(embed_dim//2)
                )
            row_head.append(cell_row)
            col_head.append(cell_col)
        self.col_head = nn.ModuleList(col_head)
        self.row_head = nn.ModuleList(row_head)

        self.init_cell_box_regression = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//2),
                nn.ReLU(),
                nn.Linear(embed_dim//2, cell_box_point_nums),
            )
        self.embed_dim = embed_dim
        # self.drop = nn.Dropout(0.1)
        embed_dim = 256
        encoder=dict(
            in_channels=[256, 512],
            feat_strides=[8, 16],
            hidden_dim=256,
            use_encoder_idx=[1],
            num_encoder_layers=1,
            pe_temperature=10000,
            expansion=1.0,
            depth_mult=1.0,
            num_heads=8,
            dropout=0,
            feedforward_dim=1024,
        )
        self.encoder = HybridEncoder(**encoder)

        positional_encoding = dict(num_feats=128,normalize=True,offset=-0.5,temperature=10000)
        self.positional_encoding = SinePositionalEncoding(**positional_encoding)
        
        self.neck = ChannelMapper(in_channels=[256, 512, 1024, 2048],
                                kernel_size=1,
                                out_channels=256,
                                act_cfg=None,
                                norm_cfg=dict(type='GN', num_groups=32),
                                num_outs=4)
        layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                cross_attn_cfg=dict(embed_dims=256, num_levels=2, dropout=0.0),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=256, ffn_drop=0.0)
                )
        l = 3
        layers = list()
        regress_head = list()
        for _ in range(l):
            layers.append(
                DeformableDetrTransformerDecoderLayer(**layer_cfg)
                )
            cell_box_regression = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(embed_dim//2, cell_box_point_nums),
                )
            regress_head.append(cell_box_regression)
        self.layers = nn.ModuleList(layers)
        self.regress_heads = nn.ModuleList(regress_head)
        
        self.ref_point_head = nn.Sequential(
                                nn.Linear(embed_dim, embed_dim//2),
                                nn.ReLU(),
                                nn.Linear(embed_dim//2, embed_dim),
                            )
        logger.info('Loading cell fine-grained information decoder ouput Done')


    def pre_transformer(
            self,
            batch_inputs,
            batch_data_samples):
        img_feats= self.neck(batch_inputs) 
        mlvl_feats = img_feats[1:-1]

        batch_size = mlvl_feats[0].size(0)
        # pdb.set_trace()
        # construct binary masks for the transformer. - 做一个原图级别的mask
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape # padding 过后的尺寸
        img_shape_list = [sample.img_shape for sample in batch_data_samples] # 原尺寸
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        # 根据特征尺度 制造多尺度特征mask,并制造PE
        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]).to(feat))
        # 特征拉直,特征/mask/PE
        # feat_flatten = []
        # lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            # if lvl in [0,1]:
            #     continue 
            batch_size, c, h, w = feat.shape
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=mask_flatten.device)
        
        spatial_shapes_i = spatial_shapes.float()
        level_start_index = torch.cat((
            spatial_shapes_i.new_zeros((1, )),  # (num_level)
            spatial_shapes_i.prod(1).cumsum(0)[:-1])).to(torch.long)
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        memory = self.encoder(
            feats=mlvl_feats,
            multi_level_pos_embeds=mlvl_pos_embeds)

        encoder_inputs_dict = dict(
            feat=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios )
        return encoder_inputs_dict
    
    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.to(torch.float16) / H
        valid_ratio_w = valid_W.to(torch.float16) / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_cell_position(self, image_embedding, cell_embedding, image_shape): # 通过交叉注意力来进行定位位置。
        cell_token = self.cell_att_fc(cell_embedding)
        img_token = self.image_att_fc(image_embedding)
        cell_position = self.init_cell_box_regression(cell_token)
        init_coord1 = F.sigmoid( cell_position ) 
        cell_token = self.cell_token_fc(cell_token)
        return init_coord1, cell_token # 需要对这个初始点进行监督
    
    def cell_regression(self, image_embed, cell_token,init_coord,pred_row,pred_col,spatial_shapes,valid_ratios,level_start_index):
        """
        使用可变性注意力来计算坐标。
        """

        reference_points = init_coord.detach()
        points_list = list()
        points_list.append(init_coord)
        
        
        row_mask = torch.where(pred_row.sigmoid()>0.5,1,0).to(pred_row)
        if torch.sum(row_mask)==0:
            row_mask = torch.eye(row_mask.shape[0])
        col_mask = torch.where(pred_col.sigmoid()>0.5,1,0).to(pred_col)
        if torch.sum(col_mask)==0:
            col_mask = torch.eye(col_mask.shape[0])


        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1]==2:
                query_sine_embed = coordinate_to_encoding(  reference_points.unsqueeze(2),num_feats=256//2)
            else:
                query_sine_embed = coordinate_to_encoding2(  reference_points.unsqueeze(2),num_feats=256//4)
            print(query_sine_embed.dtype,self.ref_point_head[0].weight.dtype)
            query_pos = self.ref_point_head(query_sine_embed.to(cell_token))
            cell_token = layer(
                cell_token,
                key=image_embed, # MSDA可注释掉这行
                key_pos=None, # MSDA可注释掉这行
                query_pos=query_pos,
                value=image_embed,
                key_padding_mask=None,
                self_attn_mask=None,
                row_same_mask=row_mask,
                col_same_mask=col_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points.unsqueeze(2),
                ) 
            cell_posi = self.regress_heads[layer_id](cell_token)
            if reference_points.shape[-1]==2:
                cell_posi[:,:,0:2] = cell_posi[:,:,0:2]+inverse_sigmoid(reference_points)
            else:
                cell_posi = cell_posi+inverse_sigmoid(reference_points)

            cell_posi = F.sigmoid(cell_posi)
            points_list.append(cell_posi)
            
            reference_points = cell_posi.detach()

        return points_list

    def forward(self, cell_hidden_state,image_embedding, image_shape,row_same_mask, col_same_mask,row_col_positions, donuts_out, gt_inst):

        init_coord,cell_token = self.init_cell_position(image_embedding, cell_hidden_state,image_shape)
        encoder_inputs_dict = self.pre_transformer(donuts_out, [gt_inst])
        cell_boxes_pred = self.cell_regression(encoder_inputs_dict["feat"], 
                                                cell_token, init_coord,
                                                row_same_mask, col_same_mask,
                                                encoder_inputs_dict["spatial_shapes"],
                                                encoder_inputs_dict["valid_ratios"],
                                                encoder_inputs_dict["level_start_index"])
        return cell_boxes_pred



    def compute_adjacency_matrix(self, input_tensor, h=4):
        """
        计算输入张量的邻接矩阵
        
        参数:
        input_tensor (torch.Tensor): 输入张量，形状为B*L*dim
        
        返回:
        torch.Tensor: 邻接矩阵，形状为B*L*L
        """
        # 通过内积计算相似度矩阵
        b,l,c = input_tensor.shape
        input_tensor_mh = input_tensor.transpose(1, 2).reshape((b,h,-1,l))
        adj_matrix = torch.matmul(input_tensor_mh.transpose(2,3), input_tensor_mh)/torch.sqrt(torch.tensor(self.embed_dim//8, dtype=torch.float32))
        adj_matrix = torch.mean(adj_matrix, dim=1)

        return adj_matrix

    
def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """

    dim_t = torch.arange( num_feats, dtype=torch.float32, device=coord_tensor.device)

    # dim_t_1 = temperature**(2 * (dim_t // 2) / num_feats)
    dim_t=temperature**(2*torch.div(dim_t ,2, rounding_mode='floor')/num_feats)

    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    # else:
    #     raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
    #         coord_tensor.size(-1)))
    return pos

def coordinate_to_encoding2(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """

    dim_t = torch.arange( num_feats, dtype=torch.float32, device=coord_tensor.device)

    # dim_t_1 = temperature**(2 * (dim_t // 2) / num_feats)
    dim_t=temperature**(2*torch.div(dim_t ,2, rounding_mode='floor')/num_feats)

    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    # if coord_tensor.size(-1) == 2:
    #     pos = torch.cat((pos_y, pos_x), dim=-1)
    # elif coord_tensor.size(-1) == 4:
    w_embed = coord_tensor[..., 2] * scale
    pos_w = w_embed[..., None] / dim_t
    pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                        dim=-1).flatten(2)

    h_embed = coord_tensor[..., 3] * scale
    pos_h = h_embed[..., None] / dim_t
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                        dim=-1).flatten(2)

    pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    # else:
    #     raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
    #         coord_tensor.size(-1)))
    return pos
def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def bbox_cxcywh_to_xyxy(bbox) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)



class IPTBboxEmbedding(nn.Module):
    def __init__(self, bbox_embed_num, bbox_embed_size, ipt_model_embed) -> None:
        super().__init__()
        
        self.bbox_embedding = nn.Embedding(bbox_embed_num, bbox_embed_size, device=torch.cuda.current_device())
        self.ipt_embedding = ipt_model_embed
        self.ipt_vocab_max = ipt_model_embed.word_embeddings.num_embeddings
        self.bbox_vocab_max = self.ipt_vocab_max + bbox_embed_num
        
        self.ocr_embedding = nn.Embedding(2, bbox_embed_size, device=torch.cuda.current_device())
        self.ocr_vocab_max = self.bbox_vocab_max + 2

        # init bbox_embedding
        mean_ipt_embed = torch.mean(ipt_model_embed.word_embeddings.weight.data, dim=0, keepdim=True)  # 1, 2560
        self.bbox_embedding.weight.data[:] = mean_ipt_embed
        self.ocr_embedding.weight.data[:] = mean_ipt_embed

        text = "\n\n\nIPTBboxEmbedding:"
        text += "\nsrc_word_embeddings: {}".format(ipt_model_embed.word_embeddings.num_embeddings)
        text += "\nbox_embed_num: {}".format(bbox_embed_num)
        text += "\nocr_embed_num: {}".format(2)
        text += "\nall_num: {}".format(self.ocr_vocab_max)
        logger.info(text)

    def get_logits_parallel(self, hidden_states):
        logits_parallel = parallel_lm_logits(
            hidden_states,
            self.ipt_embedding.word_embeddings.weight,
            parallel_output=True
        )    # B, L, 60000
        logits_bbox = torch.matmul(hidden_states.transpose(0, 1).contiguous(), self.bbox_embedding.weight.t()).transpose(0, 1).contiguous()  # B, L, 1030
        logits_ocr = torch.matmul(hidden_states.transpose(0, 1).contiguous(), self.ocr_embedding.weight.t()).transpose(0, 1).contiguous()  # B, L, 2
        logits_parallel = torch.cat([logits_parallel, logits_bbox, logits_ocr], dim=-1)
        

        return logits_parallel

    # jazhang
    def forward(self, tokens, position_ids=None):
        
        # tokens: B,L
        ipt_word_mask = tokens < self.ipt_vocab_max
        bbox_word_mask = (tokens >= self.ipt_vocab_max) * (tokens < self.bbox_vocab_max)
        ocr_word_mask = (tokens >= self.bbox_vocab_max) * (tokens < self.ocr_vocab_max)
        
        ipt_tokens = tokens.clone()
        bbox_tokens = tokens.clone()
        ocr_tokens = tokens.clone()
        
        ipt_tokens[~ipt_word_mask] = 0
        bbox_tokens[~bbox_word_mask] = self.ipt_vocab_max
        bbox_tokens -= self.ipt_vocab_max
        ocr_tokens[~ocr_word_mask] = self.bbox_vocab_max
        ocr_tokens -= self.bbox_vocab_max

        ipt_embed = self.ipt_embedding(ipt_tokens, position_ids).permute(1, 0, 2)  # L,B,D -> B,L,D
        bbox_embed = self.bbox_embedding(bbox_tokens)  # B,L,D
        ocr_embed = self.ocr_embedding(ocr_tokens)    # B,L,D
        output_embed = ipt_embed * ipt_word_mask.unsqueeze(-1) + \
                        bbox_embed * bbox_word_mask.unsqueeze(-1) + \
                        ocr_embed * ocr_word_mask.unsqueeze(-1)
        return output_embed

    def inference(self, tokens):
        # tokens: B,L
        ipt_word_mask = tokens < self.ipt_vocab_max
        bbox_word_mask = (tokens >= self.ipt_vocab_max) * (tokens < self.bbox_vocab_max)
        ocr_word_mask = (tokens >= self.bbox_vocab_max) * (tokens < self.ocr_vocab_max)
        
        ipt_tokens = tokens.clone()
        bbox_tokens = tokens.clone()
        ocr_tokens = tokens.clone()
        
        ipt_tokens[~ipt_word_mask] = 0
        bbox_tokens[~bbox_word_mask] = self.ipt_vocab_max
        bbox_tokens -= self.ipt_vocab_max
        ocr_tokens[~ocr_word_mask] = self.bbox_vocab_max
        ocr_tokens -= self.bbox_vocab_max

        ipt_embed = self.ipt_embedding.inference(ipt_tokens)  # B,L,D
        bbox_embed = self.bbox_embedding(bbox_tokens)  # B,L,D
        ocr_embed = self.ocr_embedding(ocr_tokens)    # B,L,D
        output_embed = ipt_embed * ipt_word_mask.unsqueeze(-1) + \
                        bbox_embed * bbox_word_mask.unsqueeze(-1) + \
                        ocr_embed * ocr_word_mask.unsqueeze(-1)
        return output_embed

