import segmentation_models_pytorch as smp
import torch
import torch.nn as nn



class BCEDiceLoss(nn.Module):
    def __init__(self, weight1=0.5, weight2=0.5):
        super(BCEDiceLoss, self).__init__()

        self.weight1 = weight1  
        self.weight2 = weight2  

        self.loss_fn1 = nn.BCEWithLogitsLoss()  
        self.loss_fn2 = smp.losses.DiceLoss(smp.losses.constants.MULTILABEL_MODE)          

    def forward(self, predictions, targets):
        
        loss1 = self.loss_fn1(predictions, targets)
        loss2 = self.loss_fn2(predictions, targets)
        
        
        total_loss = self.weight1 * loss1 + self.weight2 * loss2
        return total_loss