import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F
import cv2

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayInferenceDataset


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
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


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

def remove_small_objects(output, min_size=2000):
    """
    Removes small objects in a (29, W, H) boolean output based on area size.
    
    Args:
        output (numpy.ndarray): Input output array of shape (29, W, H) with boolean values.
        min_size (int): Minimum area size for objects to be retained.
        
    Returns:
        numpy.ndarray: Output array of shape (29, W, H) with small objects removed.
    """
    # Validate the input
    if output.dtype != bool:
        raise ValueError("Input output must be of boolean type.")
    
    # Create an empty array for the cleaned output
    cleaned_output = np.zeros_like(output, dtype=bool)
    
    # Process each class independently
    for i in range(output.shape[0]):
        # Convert boolean mask to uint8 for OpenCV
        binary_mask = output[i].astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Iterate through each connected component
        for j in range(1, num_labels):  # Label 0 is the background
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= min_size:  # Retain only components larger than the threshold
                cleaned_output[i][labels == j] = True
    
    return cleaned_output


def remove_small_objects(output, min_size=2000):
    """
    Removes small objects in a (29, W, H) boolean output based on area size.

    Args:
        output (numpy.ndarray): Input output array of shape (29, W, H) with boolean values.
        min_size (int): Minimum area size for objects to be retained.

    Returns:
        numpy.ndarray: Output array of shape (29, W, H) with small objects removed.
    """
    # Validate the input
    if output.dtype != bool:
        raise ValueError("Input output must be of boolean type.")

    # Create an empty array for the cleaned output
    cleaned_output = np.zeros_like(output, dtype=bool)

    # Process each class independently
    for i in range(output.shape[0]):
        # Convert boolean mask to uint8 for OpenCV
        binary_mask = output[i].astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # Iterate through each connected component
        for j in range(1, num_labels):  # Label 0 is the background
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= min_size:  # Retain only components larger than the threshold
                cleaned_output[i][labels == j] = True

    return cleaned_output


def inference(args, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model).to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for images, image_names in data_loader:
                images = images.to(device)
                outputs = model(images)

                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > args.thr).detach().cpu().numpy()

                for output, image_name in zip(outputs, image_names):
                    output = remove_small_objects(output)
                    label_slice = None
                    if data_loader.dataset.label_slice is not None:
                        label_slice = data_loader.dataset.label_slice
                    else:
                        label_slice = range(29)
                    for c, segm in zip(label_slice, output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(
                            f"{data_loader.dataset.ind2class[c]}_{image_name}"
                        )

                pbar.update(1)

    return rles, filename_and_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model to use")
    parser.add_argument(
        "--image_root", type=str, default="/data/ephemeral/home/data/test/DCM"
    )
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="./output.csv")
    parser.add_argument(
        "--resize",
        type=int,
        default=512,
        help="Size to resize images (both width and height)",
    )
    parser.add_argument(
        "--label_slice",
        type=int,
        nargs="+",
        help="List of class indices to use (e.g. --label_slice 18 19 20)",
    )
    args = parser.parse_args()

    fnames = {
        osp.relpath(osp.join(root, fname), start=args.image_root)
        for root, _, files in os.walk(args.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    tf = A.Resize(height=args.resize, width=args.resize)

    test_dataset = XRayInferenceDataset(
        fnames, args.image_root, transforms=tf, label_slice=args.label_slice
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    rles, filename_and_class = inference(args, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    df.to_csv(args.output, index=False)
