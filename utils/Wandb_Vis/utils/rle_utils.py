import numpy as np
import pandas as pd
from PIL import Image

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = map(np.asarray, (s[0::2], s[1::2]))
    # starts와 lengths를 명시적으로 정수형으로 변환
    starts = starts.astype(np.int32)
    lengths = lengths.astype(np.int32)
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(height * width, dtype=np.uint8)
    mask[np.concatenate([np.arange(start, end) for start, end in zip(starts, ends)])] = 1
    return mask.reshape((height, width))

def create_pred_mask_dict(csv_path, input_size):
    """
    CSV 파일에서 RLE 데이터를 읽어 마스크 생성
    """
    df = pd.read_csv(csv_path)
    mask_dict = {}
    grouped = df.groupby('image_name')
    for image_name, group in grouped:
        masks = {}
        for _, row in group.iterrows():
            classname = row['class']
            rle = row['rle']
            if isinstance(rle, str):
                mask = decode_rle_to_mask(rle, 2048, 2048).astype(np.uint8)
                mask_resized = np.array(Image.fromarray(mask).resize((input_size, input_size)))
                masks[classname] = mask_resized
        mask_dict[image_name] = masks
    return mask_dict
