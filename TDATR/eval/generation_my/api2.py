"""Inference API."""
import torch

from TDATR_utils.global_context import global_context as gpc

from TDATR_utils.utils import broadcast_float_list

from .generation import (
        generate_tokens_probs_and_return_on_first_stage)



def generate2(model,
             tokenizer,
             tokens = None, 
             inputs_embeds=None,
             img_embeds=None,
             inputs_embeds_length = None, 
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             temperature=1.0,
             penalty=1.7, 
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             random_seed=-1):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """
    
    
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              random_seed]
    values_float_tensor = broadcast_float_list(10, float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    add_BOS = bool(values_float_tensor[5].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[6].item())
    stop_on_double_eol = bool(values_float_tensor[7].item())
    stop_on_eol = bool(values_float_tensor[8].item())
    random_seed = int(values_float_tensor[9].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)
    

    if torch.distributed.get_rank() == 0:
        assert inputs_embeds is not None
    
    context_tokens_tensor = tokens
    context_length_tensor = inputs_embeds_length

    context_chat_length_tensor = torch.cuda.LongTensor(context_length_tensor)
    
    gen_outputs= generate_tokens_probs_and_return_on_first_stage(
        model, tokenizer, context_tokens_tensor, inputs_embeds, context_length_tensor,
        kv_hidden_states=img_embeds,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        temperature=temperature,
        penalty=penalty,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol)
    (gen_tokens, generated_sequence_lengths, output_log_probs,context_length_ywx,hidden_state_list) = gen_outputs


    


    # Only post-process on first stage.
    if gpc.is_pipeline_first_stage():
        outputs=[]
        if output_log_probs is not None:
            for (prompt_token, prompt_len, gen_token, gen_len, prob) in \
                zip(context_tokens_tensor, context_length_tensor, gen_tokens, generated_sequence_lengths, output_log_probs):
                outinfo= gen_output(prompt_token, prompt_len, gen_token, gen_len, prob, tokenizer)
                outputs.append(outinfo)
        else:
            for (prompt_token, prompt_len, gen_token, gen_len) in \
                zip(context_tokens_tensor, context_length_tensor, gen_tokens, generated_sequence_lengths):
                outinfo= gen_output(prompt_token, prompt_len, gen_token, gen_len, None, tokenizer)
                outputs.append(outinfo)

        return (outputs,hidden_state_list),context_length_ywx

    return (None,None),context_length_ywx



def gen_output(prompt_token, prompt_len, gen_token, gen_len, prob, tokenizer):
    """
        tokenizer donot always ensure detok(tok(x))=x, e.g., continuous space, tabel, return, in some tokenizer
    """
    prompt_token= prompt_token.cpu().tolist()
    gen_token= gen_token.cpu().tolist()
    prompt_token = prompt_token[:prompt_len]
    gen_token = gen_token[prompt_len:gen_len]
    prob= prob[prompt_len:gen_len] if prob else None

    


    return {
        "prompt_token":prompt_token,
        "gen_token":gen_token,
        "prompt":tokenizer.detokenize(prompt_token),
        "generate":tokenizer.detokenize(gen_token),
        "lprobs":prob
    }
