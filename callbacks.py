from typing import Any
import logging

class Callback:
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, it, *args: Any, **kwds: Any):
        if it % self.freq == 0:
            return self.do(it, *args, **kwds)

    def do(self, it ,*args: Any, **kwds: Any):
        raise NotImplementedError

class PrintCallback(Callback):
    def do(self, it, epoch_it, loss, d_loss):
        txt = '[Epoch %02d] it=%03d, loss=%.4f'% (epoch_it, it, loss)
        for key in d_loss:
            txt = txt + ", %s: %.5f"%(key, d_loss[key])
        logging.info(txt)

class VisualizeCallback(Callback):
    def __init__(self, freq, trainer, vis_loader):
        super().__init__(freq)
        self.trainer = trainer
        self.vis_loader = vis_loader
        self.vis_iter = iter(self.vis_loader)

    def do(self, *args, **kwds):
        logging.info('Visualizing')
        try:
            batch = next(self.vis_iter)
        except StopIteration:
            logging.info('Finished a loop of the visualization dataset. ')
            self.vis_iter = iter(self.vis_loader)
            batch = next(self.vis_iter)

        self.trainer.visualize(batch)

class CheckpointsaveCallback(Callback):
    def __init__(self, freq, checkpoint_io):
        super().__init__(freq)
        self.checkpoint_io = checkpoint_io

    def do(self, it, epoch_it, *args, **kwds):
        self.checkpoint_io.save_process(it=it, epoch_it=epoch_it)

class AutosaveCallback(Callback):
    def __init__(self, freq, checkpoint_io):
        super().__init__(freq)
        self.checkpoint_io = checkpoint_io

    def do(self, it, epoch_it, *args, **kwds):
        logging.info('Autosave latest checkpoint')
        self.checkpoint_io.save_latest(it=it, epoch_it=epoch_it)
        
class ValidationCallback(Callback):
    def __init__(self, freq, checkpoint_io, trainer, val_loader, writer,
                 *args, **kwds):
        super().__init__(freq)
        self.checkpoint_io = checkpoint_io
        self.trainer = trainer
        self.val_loader = val_loader
        self.writer = writer

    def do(self, it, epoch_it, *args, **kwds):
        eval_dict = self.trainer.evaluate(self.val_loader)

        for k, v in eval_dict.items():
            self.writer.add_scalar('val/%s' % k, v, it)

        self.checkpoint_io.save_if_best(eval_dict, it=it, epoch_it=epoch_it)