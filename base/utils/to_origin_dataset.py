import pandas as pd
import json
import os
import shutil


def rle_to_mask(rle_string, height=2048, width=2048):
    """RLE 문자열을 마스크로 변환"""
    if pd.isna(rle_string):
        return None
    
    rle = list(map(int, rle_string.split()))
    mask = [0] * (height * width)
    
    for i in range(0, len(rle), 2):
        start = rle[i] - 1
        length = rle[i + 1]
        for j in range(start, start + length):
            mask[j] = 1
            
    return mask

def csv_to_jsons(csv_path, output_dir):
    """CSV 파일을 이미지별 JSON 파일로 변환"""
    df = pd.read_csv(csv_path)
    
    # 이미지별로 그룹화
    grouped = df.groupby('image_name')
    
    # output 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 이미지별로 JSON 생성
    for image_name, group in grouped:
        results = {
            "filename": image_name,
            "annotations": []
        }
        
        # 어노테이션 정보 추가
        for _, ann in group.iterrows():
            # category_id = int(ann['class'].split('-')[1])
            mask = rle_to_mask(ann['rle'])
            
            if mask:
                # bbox 계산
                rows = [i // 2048 for i, v in enumerate(mask) if v]
                cols = [i % 2048 for i, v in enumerate(mask) if v]
                if rows and cols:
                    points = []
                    for r, c in zip(rows, cols):
                        points.append([c, r])
                    
                    results["annotations"].append({
                        "id": f"pseudo_id",
                        "type": "poly_seg", 
                        "attributes": {},
                        "points": points,
                        "label": ann['class']
                    })
        
        # JSON 파일 저장
        output_path = os.path.join(output_dir, f"{image_name.split('.')[0]}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f)

def organize_json_files():
    # DCM과 JSON 파일 경로 
    dcm_dir = "/data/ephemeral/home/kgs/data/test/DCM"
    json_dir = "./test_result"
    
    # DCM 디렉토리의 모든 ID 폴더 가져오기
    id_dirs = sorted([d for d in os.listdir(dcm_dir) if d.startswith('ID')])
    
    # JSON 파일 목록
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    
    # 각 ID 폴더마다 2개씩 JSON 파일 이동
    json_idx = 0
    for id_dir in id_dirs:
        # ID 폴더 경로 생성
        dst_dir = os.path.join(json_dir, id_dir)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 현재 ID 폴더에 2개의 JSON 파일 이동
        if json_idx + 2 <= len(json_files):
            for i in range(2):
                json_file = json_files[json_idx + i]
                src_path = os.path.join(json_dir, json_file)
                dst_path = os.path.join(dst_dir, json_file)
                
                # 파일 이동
                shutil.move(src_path, dst_path)
            
            json_idx += 2

if __name__ == "__main__":
    csv_to_jsons("/data/ephemeral/home/kgs/level2-cv-semanticsegmentation-cv-8-lv3/base/output.csv", "test_result")
    organize_json_files()