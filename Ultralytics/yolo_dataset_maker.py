import json
import os
import shutil
from PIL import Image

def make_yolo_dataset(original_img_path, original_json_path, output_path):
    # 이미지와 어노테이션 경로 수집
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=original_img_path)
        for root, _dirs, files in os.walk(original_img_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=original_json_path)
        for root, _dirs, files in os.walk(original_json_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    for image_path in pngs:
        # JSON 파일에서 좌표 읽어서 txt 파일로 변환
        json_path = os.path.join(original_json_path, image_path.replace('.png', '.json'))
        txt_name = os.path.basename(image_path).replace('.png', '.txt')
        txt_path = os.path.join(output_path+"/labels", txt_name)
        
        # 이미지 파일 심볼릭 링크 생성
        src_img_path = os.path.join(original_img_path, image_path)
        dst_img_path = os.path.join(output_path+"/images", os.path.basename(image_path))
        os.symlink(src_img_path, dst_img_path)
        
        if os.path.exists(json_path):
            # json 파일에서 이미지 크기 가져오기
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                img_width = json_data['metadata']['width']
                img_height = json_data['metadata']['height']
                
            with open(txt_path, 'w') as f:
                for polygon in json_data['annotations']:
                    points = polygon['points']
                    label_map = {
                        'finger-1': 0, 'finger-2': 1, 'finger-3': 2, 'finger-4': 3, 'finger-5': 4,
                        'finger-6': 5, 'finger-7': 6, 'finger-8': 7, 'finger-9': 8, 'finger-10': 9,
                        'finger-11': 10, 'finger-12': 11, 'finger-13': 12, 'finger-14': 13, 'finger-15': 14,
                        'finger-16': 15, 'finger-17': 16, 'finger-18': 17, 'finger-19': 18, 'Trapezium': 19,
                        'Trapezoid': 20, 'Capitate': 21, 'Hamate': 22, 'Scaphoid': 23, 'Lunate': 24,
                        'Triquetrum': 25, 'Pisiform': 26, 'Radius': 27, 'Ulna': 28
                    }
                    line = str(label_map[polygon['label']])
                    for point in points:
                        # 좌표를 0~1 사이의 값으로 정규화
                        normalized_x = point[0] / img_width
                        normalized_y = point[1] / img_height
                        line += f" {normalized_x:.6f} {normalized_y:.6f}"
                    f.write(line + '\n')
    
original_img_path = "/data/ephemeral/home/data/train_fold_2/DCM"
original_json_path = "/data/ephemeral/home/data/train_fold_2/outputs_json"
output_path = "/data/ephemeral/home/data/train_fold_2/yolo"

os.makedirs(output_path+"/images", exist_ok=True)
os.makedirs(output_path+"/labels", exist_ok=True)    

if __name__ == "__main__":
    make_yolo_dataset(original_img_path, original_json_path, output_path)