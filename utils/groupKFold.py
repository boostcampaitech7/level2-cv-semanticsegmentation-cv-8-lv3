import os
from sklearn.model_selection import GroupKFold
import numpy as np

def get_group_kfold(data_dir, n_splits=5, dest_dir="/data/ephemeral/home/data/"):
    """
    폴더 이름을 그룹으로 사용하여 GroupKFold를 수행하고 결과를 저장하는 함수
    
    Args:
        data_dir (str): 데이터가 있는 디렉토리 경로
        n_splits (int): fold 개수
        dest_dir (str): 결과를 저장할 디렉토리 경로
    """
    
    # 폴더명을 그룹으로 사용
    groups = []
    
    # 이미지와 어노테이션 경로 수집
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=data_dir+'/DCM')
        for root, _dirs, files in os.walk(data_dir+'/DCM')
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=data_dir+'/outputs_json')
        for root, _dirs, files in os.walk(data_dir+'/outputs_json')
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    img_dir = os.path.join(data_dir, "DCM")
    ann_dir = os.path.join(data_dir, "outputs_json")
    
    filenames = np.array(pngs)
    labelnames = np.array(jsons)
    
    # 전체 경로 생성
    image_paths = [os.path.join(img_dir, fname) for fname in filenames]
    annotation_paths = [os.path.join(ann_dir, fname) for fname in labelnames]
    
    groups = [os.path.dirname(fname) for fname in filenames]
    ys = [0 for _ in filenames]
    
    # GroupKFold 수행
    gkf = GroupKFold(n_splits=n_splits)
    
    # fold 결과 저장
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(filenames, ys, groups=groups)):
        # 각 fold별 디렉토리 생성
        train_fold_dir = os.path.join(dest_dir, f'train_fold_{fold_idx+1}')
        val_fold_dir = os.path.join(dest_dir, f'val_fold_{fold_idx+1}')
        
        print(f'Fold {fold_idx+1}:')
        print(f'  - train_fold_{fold_idx+1}: {len(train_idx)}개')
        print(f'  - val_fold_{fold_idx+1}: {len(val_idx)}개')
        
        # 폴더가 존재하지 않을 경우 생성
        for fold_dir in [train_fold_dir, val_fold_dir]:
            os.makedirs(os.path.join(fold_dir, 'DCM'), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'outputs_json'), exist_ok=True)
        
        # train 데이터 심볼릭 링크 생성
        for idx in train_idx:
            img_name = os.path.basename(image_paths[idx])
            ann_name = os.path.basename(annotation_paths[idx])
            
            os.symlink(image_paths[idx], 
                      os.path.join(train_fold_dir, 'DCM', img_name))
            os.symlink(annotation_paths[idx], 
                      os.path.join(train_fold_dir, 'outputs_json', ann_name))
        
        # validation 데이터 심볼릭 링크 생성
        for idx in val_idx:
            img_name = os.path.basename(image_paths[idx])
            ann_name = os.path.basename(annotation_paths[idx])
            
            os.symlink(image_paths[idx], 
                      os.path.join(val_fold_dir, 'DCM', img_name))
            os.symlink(annotation_paths[idx], 
                      os.path.join(val_fold_dir, 'outputs_json', ann_name))

if __name__ == "__main__":
    data_dir = "/data/ephemeral/home/data/train/"
    fold_indices = get_group_kfold(data_dir, n_splits=5)
    print(fold_indices)