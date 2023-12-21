from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class EMATrainHook(Hook):
    r"""Exponential Moving Average Hook.
    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below.
        .. math::
            Xema\_{t+1} = (1 - \text{momentum}) \times
            Xema\_{t} +  \text{momentum} \times X_t
    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str, optional): The checkpoint path. Defaults to None.
    """

    def __init__(self,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 warm_up: int = 100):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum**interval

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.
        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        # collect teacher parameters and corresponding student ones
        model_parameters = dict(model.named_parameters(recurse=True))
        self.teacher_parameters = dict()
        self.student_parameters = dict()
        for key in model_parameters.keys():
            if key.startswith("teacher_"):
                student_key = key[len("teacher_"):]
                self.teacher_parameters[key] = model_parameters[key] 
                self.student_parameters[student_key] = model_parameters[
                    student_key] 

        # collect teacher buffers and corresponding student ones if from bn
        model_buffers = dict(model.named_buffers(recurse=True))
        self.teacher_buffers = dict()
        self.student_buffers = dict()
        for key in model_buffers.keys():
            if key.startswith("teacher_") and "bn" in key and "running" in key:
                student_key = key[len("teacher_"):]
                self.teacher_buffers[key] = model_buffers[key] 
                self.student_buffers[student_key] = model_buffers[
                    student_key] 

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        # ema update to teacher parameters 
        for key, teacher_parameter in self.teacher_parameters.items():
            assert key.startswith("teacher_")
            assert teacher_parameter.requires_grad == False
            student_key = key[len("teacher_"):]
            student_parameter = self.student_parameters[student_key] 
            teacher_parameter.mul_(1 - momentum).add_(
                student_parameter.data, alpha=momentum)

        # ema update to teacher buffers
        for key, teacher_buffer in self.teacher_buffers.items():
            assert key.startswith("teacher_")
            student_key = key[len("teacher_"):]
            student_buffer = self.student_buffers[student_key] 
            teacher_buffer.mul_(1 - momentum).add_(
                student_buffer.data, alpha=momentum)
