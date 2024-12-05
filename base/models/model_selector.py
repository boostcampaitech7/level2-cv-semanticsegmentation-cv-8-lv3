from .base_model import UnetModel
import segmentation_models_pytorch as smp


class ModelSelector:
    """
    model을 새롭게 추가하기 위한 방법
        1. model 폴더 내부에 사용하고자하는 custom model 구현
        2. 구현한 Model Class를 model_selector.py 내부로 import
        3. self.model_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 model_name을 설정한 key값으로 변경
    """

    def __init__(self) -> None:
        self.model_classes = {
            "Unet": UnetModel,
        }

    def get_model(self, model_name, **model_parameter):
        if "label_slice" in model_parameter:
            # Adjust number of classes based on label_slice
            num_classes = (
                len(model_parameter["label_slice"])
                if model_parameter["label_slice"]
                else 29
            )
            model_parameter["classes"] = num_classes
            del model_parameter["label_slice"]
        model = smp.create_model(**model_parameter)
        return model
