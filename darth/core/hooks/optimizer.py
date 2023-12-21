from mmcv.runner.hooks import HOOKS, OptimizerHook


@HOOKS.register_module()
class CustomOptimizerHook(OptimizerHook):
    """This class wraps the original optimizer hook to allow skipping backward
    pass when the total loss does not have any gradient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backward_exception_thrown = False

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        try:
            runner.outputs['loss'].backward()
        except:
            if not self.backward_exception_thrown:
                print(
                    'Backward pass failed. Check if your loss has gradients. '
                    'This warning will be thrown only once.')
                self.backward_exception_thrown = True

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        runner.outputs['num_samples'])
        runner.optimizer.step()
