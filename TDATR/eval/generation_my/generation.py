"""Generation utilities."""

import torch
import torch.nn.functional as F
from typing import List, Iterable
from transformers import LogitsProcessor
from TDATR_utils.forward_step import ForwardStep
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.utils import copy_from_last_to_first_pipeline_stage, broadcast_from_last_pipeline_stage,broadcast_from_last_to_first_pipeline_stage, sample


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1

    attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data).clone()
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        attention_mask =  torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    if attention_mask != None:
        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, input_lens):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx][input_lens[idx]:].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int, input_lens
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos, input_lens)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessorRightPad(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, input_lens) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len, input_lens)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float, num_penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.num_penalty = num_penalty
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, input_lens) -> torch.FloatTensor:
        current_len = input_ids.size(1)
        # (batch_size, current_len)
        # index = 0
        # for ids, length in zip(input_ids, input_lens):
        #     id_each, num = torch.unique(ids[length:],return_counts=True)
        #     scores[index, id_each] -= self.num_penalty * (num - 1)
        #     index += 1
        # pdb.set_trace()
        score = torch.gather(scores, 1, input_ids)
        # ori_score = score.clone()
        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        # score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        # score_old = score
        score -= self.penalty
        # print(score)
        # score_num = []
        # for ind, each in enumerate(score):
        #     for ind_col, ea in enumerate(each[input_lens[ind]:]):
        #         # print(sum(each == ea))
        #         if sum(each[input_lens[ind]:] == ea) > 1:
        #             score[ind, ind_col] -= (self.num_penalty * (sum(each[input_lens[ind]:] == ea) - 1))

        # print(len(input_lens))
        # print(score_num)
        # print(input_lens)

        for t in range(len(input_lens)):
            update_score = score[t, input_lens[t]:]

            answer_ids = input_ids[t, input_lens[t]:]
            scores[t].scatter_(0, answer_ids, update_score)
        # mask = torch.ones(len(input_lens), current_len).to(input_ids)
        # for t in range(len(input_lens)):
        #     mask[t, : input_lens[t]] = 0
        # update_score = score * mask + (1 - mask) * ori_score
        # scores.scatter_(1, input_ids, update_score)
        # print(torch.gather(scores, 1, input_ids))
        # print(torch.gather(scores, 1, input_ids) - score_old)
        return scores

        
class MinLengthLogitsProcessorRightPad(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, input_lens) -> torch.FloatTensor:
        batch_len = input_ids.shape[-1]
        num_batch_hypotheses = scores.shape[0]
        for idx in range(num_batch_hypotheses):
            cur_len = batch_len - input_lens[idx]
            if cur_len < self.min_length:
                scores[idx, self.eos_token_id] = -float("inf")
        return scores


def generate_tokens_probs_and_return_on_first_stage(
        model, tokenizer, tokens, embeds, lengths,
        kv_hidden_states=None,
        return_output_log_probs=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        penalty=1.7,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False
):
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        embeds: embeddings of tokens [b, max-sequence-length, embedding_dim]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)
    max_sequence_length = min(max_sequence_length, gpc.config.model.max_position_embeddings)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.

    termination_id = tokenizer.eos_token_id

    # no_repeat_ngram_logits_processor = NoRepeatNGramLogitsProcessorRightPad(gpc.config.generation.no_repeat_ngram_size)
    no_repeat_ngram_logits_processor = RepetitionPenaltyLogitsProcessor(penalty, 0.0)
    min_length_logits_processor = MinLengthLogitsProcessorRightPad(gpc.config.generation.min_len, termination_id)

    # If the context is too big, this happens
    if min_prompt_length >= max_sequence_length:
        raise ValueError("context length + tokens_to_generate too large")

    # forward step.
    forward_step = ForwardStep(model.ipt_model, batch_size, max_sequence_length)

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if gpc.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length
    
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())
    
    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens) #attention_mask == None, [0,1,...,4112(16+4096)]
        
        ##### modified by wcliu4, generate attention_mask for minigpt4
        if gpc.config.model.get('use_attn_mask', False):
            B, L = tokens.shape
            causal_mask = torch.tril(torch.ones(B, L, L))
            local_mask = torch.zeros(B, L, L)
            valid_idxs = torch.cat([torch.ones([B, min_prompt_length]), 
                                    torch.zeros([B, max_sequence_length-min_prompt_length])], dim=1)
            local_mask[valid_idxs==1, :] = 1
            local_mask = local_mask * local_mask.permute(0, 2, 1)
            attention_mask = (1 - local_mask) * (1 - causal_mask)
            attention_mask = attention_mask.to(tokens.device).to(torch.bool)
        else:
            attention_mask = None   #-> this step
        ##############################
        
        prev_context_length = 0
        context_length = None
        # input_lens = torch.sum(tokens != 2, dim=-1).cpu().tolist()
        input_lens = lengths.cpu().tolist()
        hidden_state_list = list()

        for context_length in range(min_prompt_length, max_sequence_length):
            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            embeds2use = embeds[:, prev_context_length:context_length, :]
            positions2use = position_ids[:, prev_context_length:context_length]
            if attention_mask is not None:
                attention_mask2use = attention_mask[
                    ..., prev_context_length:context_length, :context_length]
            else:
                attention_mask2use = None


            # logits will be meanigful only in the last pipeline stage.
            # import pdb
            # 将 hidden_list 包裹在 hidden_states,
            logits, (hidden_states, hidden_state_list_temp) = forward_step(embeds2use, kv_hidden_states, positions2use, attention_mask2use, return_hidden_states=True)
            # print([i.shape for i in hidden_state_list_temp])
            hidden_state_list.append(torch.stack(hidden_state_list_temp, dim=-1))
            logits = model.ipt_bbox_embedding.get_logits_parallel(hidden_states)
            # print("hidden_states.shape",hidden_states.shape)
            # print("hidden_state_list[0].shape",hidden_state_list[0].shape)
            # print()
            ## 解码 token 进行ngram、长度等约束处理
            next_token_logits = logits[:, -1, :]
            if gpc.config.generation.no_repeat_ngram_size >= 0:
                next_token_logits = no_repeat_ngram_logits_processor(tokens[:, :context_length], next_token_logits, input_lens)
            if gpc.config.generation.min_len > 1:
                next_token_logits = min_length_logits_processor(tokens[:, :context_length], next_token_logits, input_lens)

            new_sample, started = None, None
            if gpc.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                # last_token_logits = logits[:, -1, :]
                last_token_logits = next_token_logits
                new_sample = sample(last_token_logits,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    vocab_size=tokenizer.vocab_size)
                
                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]
                # Update the embeddings
                # pdb.set_trace()
                embeds[started, context_length] = model.ipt_bbox_embedding(new_sample[started].unsqueeze(0), None).squeeze(0)

                # Calculate the log probabilities.
                if return_output_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    if return_output_log_probs:
                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        indices = torch.unsqueeze(
                            tokens[:, (prev_context_length + 1):(context_length + 1)],
                            dim = 2,
                        )
                        output_log_probs[:, prev_context_length:context_length] = \
                            torch.gather(log_probs, 2, indices).squeeze(2)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])
            copy_from_last_to_first_pipeline_stage((batch_size, embeds.shape[-1]), embeds.dtype, embeds[:, context_length, :])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if gpc.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                if stop_on_double_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_two_eols
                elif stop_on_eol:
                    hit_double_eol = (new_sample == 628).byte() & started.byte()
                    hit_eol = (new_sample == 198).byte() & started.byte()
                    done_token = hit_double_eol | hit_eol
                else: 
                    done_token = (new_sample == termination_id).byte() & \
                        started.byte()
                
                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_eod_token_for_early_termination and done:
                break
            
    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if gpc.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]
            if not output_log_probs.is_contiguous():
                output_log_probs = output_log_probs.contiguous()

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    return tokens, generated_sequence_lengths, output_log_probs, context_length, hidden_state_list


def _build_attention_mask_and_position_ids(tokens):
    """Build the attention mask and postition ids for the input tokens."""
    
    # Since we are not interested in loss-mask and reset attention/position
    # is also False, eod_token is not used so it is safe to set it to None.
    # attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
    #     data=tokens,
    #     eod_token=None,
    #     reset_position_ids=False,
    #     reset_attention_mask=False,
    #     eod_mask_loss=False,
    #     prompt_length_list=[],
    #     eot_length_list=[],
    #     prompt_mask=False
    # )
    
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False
    )

    return attention_mask, position_ids
