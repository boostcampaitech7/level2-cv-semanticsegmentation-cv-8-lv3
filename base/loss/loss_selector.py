from .base_loss import CustomBCEWithLogitsLoss
from .bce_dice_loss import BCEDiceLoss
import segmentation_models_pytorch as smp

def jaccard_loss():
    return smp.losses.JaccardLoss(smp.losses.constants.MULTILABEL_MODE)

def dice_loss():
    return smp.losses.DiceLoss(smp.losses.constants.MULTILABEL_MODE)

def tversky_loss():
    return smp.losses.TverskyLoss(smp.losses.constants.MULTILABEL_MODE)

def focal_loss():
    return smp.losses.FocalLoss(smp.losses.constants.MULTILABEL_MODE)

def mcc_loss():
    return smp.losses.MCCLoss(smp.losses.constants.MULTILABEL_MODE)


class LossSelector():
    """
    loss를 새롭게 추가하기 위한 방법
        1. loss 폴더 내부에 사용하고자하는 custom loss 구현
        2. 구현한 Loss Class를 loss_selector.py 내부로 import
        3. self.loss_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 loss_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.loss_classes = {
            "BCEWithLogitsLoss" : CustomBCEWithLogitsLoss,
            "DiceLoss" : dice_loss,
            'JaccardLoss' : jaccard_loss,
            "TverskyLoss" : tversky_loss,
            "FocalLoss" : focal_loss,
            "MccLoss" : mcc_loss,
            "BCEDiceLoss" : BCEDiceLoss
        }
    
    def get_loss(self, loss_name, **loss_parameter):
        return self.loss_classes.get(loss_name, None)(**loss_parameter)