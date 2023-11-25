from functools import partial
import torch

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, 
                                               *, 
                                               num_warmup_steps: int, 
                                               num_decay_steps: int,
                                               initial_lr: float,
                                               peak_lr: float,
                                               final_lr: float) -> float:
    """Note that this needs to return a multiplier on `initial_lr` as set in the optimizer"""
    if current_step < num_warmup_steps:
        # Linear warmup from `initial_lr` to `peak_lr`
        new_lr: float = (peak_lr - initial_lr) / num_warmup_steps * current_step + initial_lr
    elif current_step < num_warmup_steps + num_decay_steps:
        # Linear decay from `peak_lr` to `final_lr`
        new_lr: float = (final_lr - peak_lr) / num_decay_steps * (current_step - num_warmup_steps) + peak_lr
    else:
        # Plateau at `final_lr`
        new_lr: float = final_lr
    multipler: float = new_lr / peak_lr
    return multipler

def lr_warmup_with_constant_plateau(optimizer, 
                                    num_warmup_steps: int, 
                                    num_decay_steps: int,
                                    initial_lr: float,
                                    final_lr: float, 
                                    last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases linearly from the peak lr set in the optimizer to `final_lr`, after
    a warmup period during which it increases linearly from `initial_lr` to the peak lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The total number of steps to decay the lr.
        initial_lr (`float`):
            The initial learning rate before the warmup phase
        final_lr (`float`):
            The final learning rate after the warmup and decay phases that we plateau at
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert num_warmup_steps > 0, f"num_warmup_steps must be > 0, got {num_warmup_steps}"
    assert num_decay_steps > 0, f"num_decay_steps must be > 0, got {num_decay_steps}"

    peak_lr: float = optimizer.param_groups[0]['lr']
    
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)