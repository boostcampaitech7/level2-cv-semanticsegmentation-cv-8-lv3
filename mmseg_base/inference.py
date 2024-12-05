# Python built-in
import argparse
from collections import defaultdict

# Third party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# MMEngine
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

# Local
from constants import *
from evaluator import *
from models import *
from process_data import *

IMAGE_ROOT = "/data/ephemeral/home/data/test/DCM/"


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def _preprare_data(cfg, imgs, model):
    for t in cfg.test_pipeline:
        if t.get("type") in ["LoadXRayAnnotations", "TransposeAnnotations"]:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]["type"] = "LoadImageFromNDArray"

    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data["inputs"].append(data_["inputs"])
        data["data_samples"].append(data_["data_samples"])

    return data, is_batch


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def test(cfg, model, image_paths, thr=0.5):
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            img = cv2.imread(os.path.join(IMAGE_ROOT, image_path))

            # prepare data
            data, is_batch = _preprare_data(cfg, img, model)

            # forward the model
            with torch.no_grad():
                outputs = model.test_step(data)

            outputs = outputs[0].pred_sem_seg.data
            outputs = outputs[None]

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            output = outputs[0]
            image_name = os.path.basename(image_path)
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="config file path")
    parser.add_argument(
        "--checkpoint", help="checkpoint path", default="iter_16000.pth"
    )
    parser.add_argument(
        "--csv_path", help="output csv file path", default="submission.csv"
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)
    model = MODELS.build(cfg.model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    rles, filename_and_class = test(cfg, model, pngs)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    df = pd.DataFrame(
        {
            "image_name": filename,
            "class": classes,
            "rle": rles,
        }
    )

    df.to_csv(args.csv_path, index=False)


if __name__ == "__main__":
    main()
