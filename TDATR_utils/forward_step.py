"""Forward step utilities."""
from typing import Iterable, Tuple, Union, Optional, Dict
import torch
from TDATR_utils.global_context import global_context as gpc


class InferenceParams(object):
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size: int, max_sequence_len: int, cache_enabled: bool=True):
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len: int = max_sequence_len
        self.max_batch_size: int = max_batch_size
        self.sequence_len_offset: int = 0
        self.batch_size_offset: int = 0
        self.key_value_memory_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.valid_batch_ids: Optional[torch.LongTensor]  = None
        local_att_size = None
        if gpc.config.generation.local_attention_memory_enable:
            local_att_size = getattr(gpc.config.model, "sparse_local_size", None)
        self.local_att_size: int = local_att_size
        self.cache_enabled: bool = cache_enabled  # NOTE don't need create cache when only encoding

    def set_valid_batches(self, batch_ids: torch.LongTensor) -> None:
        self.valid_batch_ids = batch_ids

    def swap_key_value_dict(self, batch_idx: int) -> None:
        "swap between batches"
        if not self.cache_enabled: # NOTE don't need create cache when only encoding
            return

        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")
        
        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1] # make sure batch size is the same
            if inference_key_memory.device.type == 'npu':
                new_inference_key_memory = inference_key_memory[:].index_select(1, batch_idx.to(inference_key_memory.device))
                new_inference_value_memory = inference_value_memory[:].index_select(1, batch_idx.to(inference_value_memory.device))
            else:
                new_inference_key_memory = inference_key_memory[:, batch_idx]
                new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)

    def create_kv_memory(self,
                         layer_number: int,
                         hidden_size: int,
                         dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cache_enabled: # NOTE don't need create cache when only encoding
            return
        if layer_number not in self.key_value_memory_dict:
            inf_max_seq_len = (
                self.local_att_size
                if self.local_att_size is not None
                else self.max_sequence_len
            )
            inf_max_batch_size = self.max_batch_size
            k_memory = self.allocate_memory(inf_max_seq_len, inf_max_batch_size, hidden_size, dtype)
            v_memory = self.allocate_memory(inf_max_seq_len, inf_max_batch_size, hidden_size, dtype)
            self.key_value_memory_dict[layer_number] = (k_memory, v_memory)
        return self.key_value_memory_dict[layer_number]

    def get_kv_memory(self, layer_number: int) -> Union[None, Tuple[torch.Tensor, torch.Tensor]]:
        return self.key_value_memory_dict.get(layer_number, None)

    def update_kv_memory(self,
                         layer_number: int,
                         k: torch.Tensor,
                         v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """concat k/v context with histories, then update histories"""
        if not self.cache_enabled: # NOTE don't need create cache when only encoding
            return k, v
        k_memory, v_memory = self.key_value_memory_dict[layer_number]
        batch_start = self.batch_size_offset
        batch_end = batch_start + k.size(1)
        assert batch_end <= k_memory.size(1)
        sequence_start = self.sequence_len_offset
        sequence_end = sequence_start + k.size(0)
        k_seqlen = k.size(0)
        local_size = self.local_att_size
        batch_ids = (
            self.valid_batch_ids[batch_start:batch_end]
            if self.valid_batch_ids is not None
            else torch.arange(batch_start, batch_end)
        )
        # if current context length is larger than local att window
        if local_size is not None and sequence_end > local_size:
            # encode is done, start to generate
            # drop out `k_seqlen` positions earliest context and cat the new context
            if k_seqlen < local_size:
                # shift the k/v context `k_seqlen` positions from right to left,
                # vacating a space of `k_seqlen` positions for new context,
                # then append new context to the memory on right.
                k_memory[:, batch_ids, ...] = torch.roll(k_memory[:, batch_ids, ...], -k_seqlen, 0)
                k_memory[local_size-k_seqlen: local_size, batch_ids, ...] = k

                v_memory[:, batch_ids, ...] = torch.roll(v_memory[:, batch_ids, ...], -k_seqlen, 0)
                v_memory[local_size-k_seqlen: local_size, batch_ids, ...] = v

                k = k_memory[:local_size, batch_ids, ...]
                v = v_memory[:local_size, batch_ids, ...]
            # encode length is longer than window size, only store right context to the memory
            else:
                k_memory[:local_size, batch_ids, ...] = k[k_seqlen-local_size: k_seqlen, ...]
                v_memory[:local_size, batch_ids, ...] = v[k_seqlen-local_size: k_seqlen, ...]
        else:
            k_memory[sequence_start: sequence_end, batch_ids, ...] = k
            v_memory[sequence_start: sequence_end, batch_ids, ...] = v
            if k_memory.device.type == 'npu':
                k = k_memory[:sequence_end].index_select(1, batch_ids.to(k_memory.device))
                v = v_memory[:sequence_end].index_select(1, batch_ids.to(k_memory.device))
            else:
                k = k_memory[:sequence_end, batch_ids, ...]
                v = v_memory[:sequence_end, batch_ids, ...]
        return k, v

    @staticmethod
    def allocate_memory(s: int, b: int, h: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(s, b, h, dtype=dtype, device=torch.cuda.current_device())



class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model: torch.nn.Module, max_batch_size: int, max_sequence_len: int, cache_enabled: bool=True):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model: torch.nn.Module = model
        # Initialize inference parameters.
        self.inference_params: InferenceParams = InferenceParams(max_batch_size, max_sequence_len, cache_enabled)
        self.pipelining_batch_x_seqlen: int = \
            gpc.config.generation.inference_batch_times_seqlen_threshold
    
    def set_valid_batches(self, valid_batches: torch.LongTensor) -> None:
        self.inference_params.set_valid_batches(valid_batches)
        # output = forward_step(tokens, position_ids, attention_mask, return_hidden_states=True)

    def __call__(self, tokens, kv_tokens, position_ids, attention_mask, return_hidden_states=False, task='layout'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           kv_tokens, # @@@ cross attention 
                                           position_ids,
                                           attention_mask,
                                           self.inference_params,
                                           return_hidden_states=return_hidden_states,
                                           task=task)


def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if gpc.is_pipeline_first_stage():
        return None
    recv_size = (sequence_length, batch_size, gpc.config.model.embed_dim)
    if gpc.config.common.fp16:
        dtype = torch.float16
    elif gpc.config.common.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return torch.empty(recv_size,
                       dtype=dtype,
                       device=torch.cuda.current_device())


def _forward_step_helper(model, tokens, kv_tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None, return_hidden_states=False, task='layout'):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params, 
                          return_hidden_states=return_hidden_states, 
                          task=task, kv_hidden_states=kv_tokens)

    return output_tensor


def _no_pipelining_forward_step(model, tokens, kv_tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None, return_hidden_states=False, task='layout'):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, kv_tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer, return_hidden_states=return_hidden_states, task=task)
    # Update the sequence length offset.
    inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if gpc.is_pipeline_last_stage():
        if return_hidden_states:
            logits, hidden_states = output_tensor
            return logits, hidden_states
        else:
            logits = output_tensor

    return logits
