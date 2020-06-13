# Helper utility function for AlexNet training.

# Importing required libraries
import torch
import shutil

# Definition of Utils class containing functions to assess mode, create checkpoints doing training etc.
class Utils(object):

    def __init__(self):
        """
        Resets 'object' on (re)initialization.
        """
        self.reset()

    def reset(self):
        """
        Resets parameters.
        """
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Take one 'update' step.
        """
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count
    
def accuracy(y_hat, y, top_k=(1, )):
    """
    Calculates model's accuracy (and top_k class accuracy) given prediction and true label.

    params
    y_hat : predicted value
    y : true label
    top_k : hols top-K accuracy

    returns
    accuracy
    """
    max_k = max(top_k)
    batch_size = y.size(0)
    _, pred = y_hat.top_k(max_k, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    acc = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc.append(correct_k._mul(100.0 / batch_size))
    return acc

def create_checkpoint(curr_state, is_best, fname='model_checkpoint.pth'):
    """
    Create model checkpoints during training. Save the best.

    params
    curr_state : current training state of the model
    is_best : Boolean, True if model's accuracy is current best
    """
    torch.save(curr_state)
    if is_best: shutil.copyfile(fname, 'alexnet_best_model.pth')

def custom_weight_decay(optimizer, epoch, init_lr):
    """
    Custom implementation of weight decay for the optimising process
    This sets the LR to the initial LR decreseas / decayed by factor of 10 every 30 epochs.
    
    params
    optimizer : torch.optimizer object
    epoch : current epoch
    init_lr : starting learning rate
    """
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups: param_group['lr'] = lr
        
