import os
import argparse
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO 모델 추론 및 결과 저장')
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 YOLO 모델의 경로 (예: runs/segment/train/weights/best.pt)')
    parser.add_argument('--test_dir', type=str, default='/data/ephemeral/home/data/test/DCM',
                        help='테스트 이미지가 있는 디렉토리 경로')
    parser.add_argument('--save_path', type=str, required=True,
                        help='결과를 저장할 CSV 파일 경로 (예: results.csv)')
    return parser.parse_args()


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def run_inference(model_path, test_dir, save_path):
    # YOLO 모델 로드
    model = YOLO(model_path)
    
    # 클래스 이름 매핑
    class_names = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
    ]
    
    # 테스트 이미지 목록 가져오기
    test_images = []
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    test_images.append(os.path.join(folder, img_name))

    # 결과를 저장할 리스트
    results_list = []
    
    # 각 이미지에 대해 추론 실행
    for img_path_rel in tqdm(test_images):
        img_path = os.path.join(test_dir, img_path_rel)
        
        # 이미지 크기 가져오기
        import cv2
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # 추론 실행 (이미지 크기 유지)
        results = model.predict(img_path, imgsz=(img_height, img_width))
        
        # 결과 처리
        for result in results:
            masks = result.masks
            if masks is not None:
                # 각 클래스별 결과 저장
                for class_name in class_names:
                    class_found = False
                    for i, mask in enumerate(masks):
                        if class_name == class_names[int(result.boxes.cls[i])]:
                            # 마스크를 numpy 배열로 변환
                            mask_array = mask.data.cpu().numpy().squeeze()
                            # RLE 인코딩
                            rle = encode_mask_to_rle(mask_array)
                            class_found = True
                            results_list.append({
                                'image_name': os.path.basename(img_path_rel),
                                'class': class_name,
                                'rle': rle
                            })
                            break
                    
                    if not class_found:
                        results_list.append({
                            'image_name': os.path.basename(img_path_rel),
                            'class': class_name,
                            'rle': ''
                        })
    
    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(results_list)
    df = df.sort_values(['image_name', 'class'])
    df['class'] = pd.Categorical(df['class'], categories=class_names)
    df = df.sort_values(['image_name', 'class'])
    df.to_csv(save_path, index=False)
    print(f"RLE 인코딩된 추론 결과가 {save_path}에 저장되었습니다.")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model_path, args.test_dir, args.save_path)
