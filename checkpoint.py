import torch
import os

class Checkpoint:
    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
    
    def __call__(self, loss, model, epoch): 
        self.save_checkpoint(loss, model, epoch)

    def save_checkpoint(self, loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        checkpoint_path = f'epoch-{epoch}-{loss}.pth'
        checkpoint_dir = os.path.join(self.exp_dir, checkpoint_path)
        torch.save(model.state_dict(), checkpoint_dir)
        self.loss_min = loss