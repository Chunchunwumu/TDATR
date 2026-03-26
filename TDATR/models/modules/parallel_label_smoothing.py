import torch
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.utils import VocabUtility

class _VocabParallelLabelSmoothing(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, epsilon):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=gpc.get_group(ParallelMode.TENSOR))
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = gpc.get_local_rank(ParallelMode.TENSOR)
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=gpc.get_group(ParallelMode.TENSOR))

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=gpc.get_group(ParallelMode.TENSOR))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        log_sum_exp_logits=  torch.log(sum_exp_logits)
        nll_loss = log_sum_exp_logits - predicted_logits

        # calculate label smoothing : the sum of logits for all
        global_vocab_size= partition_vocab_size*world_size
        eps_i = epsilon/(global_vocab_size -1)
        smooth_loss= (log_sum_exp_logits.unsqueeze(-1)-logits_2d).sum(-1)
        torch.distributed.all_reduce(smooth_loss,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=gpc.get_group(ParallelMode.TENSOR))
        
        loss= (1.0-epsilon- eps_i)* nll_loss + eps_i *smooth_loss



        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, epsilon, eps_i)

        return loss, nll_loss

    @staticmethod
    def backward(ctx, grad_output, *args):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d, epsilon, eps_i = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        tgt_smooth = eps_i
        tgt_label= 1.0-epsilon - eps_i
        grad_2d -= tgt_smooth
        grad_2d[arange_1d, masked_target_1d] -= tgt_label*(
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_label_smoothing(vocab_parallel_logits, target, label_smoothing=0.05):
    """Helper function for the cross entropy."""
    return _VocabParallelLabelSmoothing.apply(vocab_parallel_logits, target,label_smoothing)