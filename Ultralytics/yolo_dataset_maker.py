import json
import os
import shutil

original_img_path = "/data/ephemeral/home/data/train/DCM"
original_json_path = "/data/ephemeral/home/data/train/outputs_json"
output_path = "/data/ephemeral/home/data/train/yolo"

os.makedirs(output_path+"/images", exist_ok=True)
os.makedirs(output_path+"/labels", exist_ok=True)

def make_yolo_dataset(original_img_path, original_json_path, output_path):
    for patient_id in os.listdir(original_img_path):
        patient_path = os.path.join(original_img_path, patient_id)
        json_patient_path = os.path.join(original_json_path, patient_id)
        
        for image_id in os.listdir(patient_path):
            # 이미지 파일 심볼릭 링크 생성
            image_path = os.path.join(patient_path, image_id)
            os.symlink(image_path, os.path.join(output_path+"/images", f"{image_id}"))
            
            # JSON 파일에서 좌표 읽어서 txt 파일로 변환
            json_name = image_id.replace('.png', '.json')
            json_path = os.path.join(json_patient_path, json_name)
            txt_name = image_id.replace('.png', '.txt')
            txt_path = os.path.join(output_path+"/labels", txt_name)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    
                with open(txt_path, 'w') as f:
                    for polygon in json_data['annotations']:
                        points = polygon['points']
                        # 클래스 인덱스는 0으로 고정 (단일 클래스)
                        line = polygon['label']
                        for point in points:
                            line += f" {point[0]} {point[1]}"
                        f.write(line + '\n')
    
if __name__ == "__main__":
    make_yolo_dataset(original_img_path, original_json_path, output_path)