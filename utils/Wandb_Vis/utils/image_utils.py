import numpy as np
from PIL import Image

def ready_for_visualize(image, label):
    """
    이미지를 시각화에 적합한 형태로 변환
    """
    lbl = label.numpy().astype(np.uint8)
    img = image.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img, lbl
