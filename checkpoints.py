import os
import urllib
import torch
from torch.utils import model_zoo
import logging

class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''
    def __init__(self, model, optimizer=None, lr_scheduler=None, checkpoint_dir='./chkpts'):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def set_selection_criteria(self, model_selection_metric, model_selection_sign, metric_val_best=None):
        self.model_selection_metric = model_selection_metric
        self.model_selection_sign = model_selection_sign
        self.metric_val_best = metric_val_best

    def save_if_best(self, eval_dict, it, epoch_it, **kwargs):
        metric_val = eval_dict[self.model_selection_metric]
        logging.info('Validation metric (%s): %.4f'
            % (self.model_selection_metric, metric_val))

        if self.model_selection_sign * (metric_val - self.metric_val_best) > 0:
            self.metric_val_best = metric_val
            logging.info('New best model (loss %.4f)' % self.metric_val_best)
            self.save('model_best.pt', loss_val_best=self.metric_val_best, it=it, epoch_it=epoch_it, **kwargs)

    def save_latest(self, it, epoch_it, **kwargs):
        logging.info('Saving checkpoint model.pt')
        self.save('model.pt', loss_val_best=self.metric_val_best, it=it, epoch_it=epoch_it, **kwargs)

    def save_process(self, it, epoch_it, **kwargs):
        logging.info('Backup checkpoint model_%d.pt'%it)
        self.save('model_%d.pt'%it, loss_val_best=self.metric_val_best, it=it, epoch_it=epoch_it, **kwargs)
    # def register_modules(self, **kwargs):
    #     ''' Registers modules in current module dictionary.
    #     '''
    #     self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        outdict['model'] = self.model.state_dict()
        if self.optimizer is not None:
            outdict['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            outdict['lr_scheduler'] = self.lr_scheduler.state_dict()
            
        torch.save(outdict, filename)

    def load(self, filename):
        '''Loads a module dictionary from local file or url.
        
        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename)

    def load_file(self, filename):
        '''Loads a module dictionary from file.
        
        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
            scalars = self.load_state_dict(state_dict)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url):
        '''Load a module dictionary from url.
        
        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.load_state_dict(state_dict)
        return scalars

    def load_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.
        
        Args:
            state_dict (dict): State dict of model
    '''
        self.model.load_state_dict(state_dict.pop('model'))
        if self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict.pop('optimizer'))
            except Exception as e:
                logging.warn('Cannot find optimizer in checkpoint: {}'.format(e))
        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.load_state_dict(state_dict.pop('lr_scheduler'))
            except Exception as e:
                logging.warn('Cannot find lr_scheduler in checkpoint: {}'.format(e))
                
        return state_dict

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')