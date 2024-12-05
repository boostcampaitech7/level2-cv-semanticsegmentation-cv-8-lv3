import numpy as np # numpy import
import pandas as pd # pandas import
import os # os import
from tqdm import tqdm # 진행상황 확인을 위한 tqdm import

def csv_ensemble(csv_paths, save_dir, threshold):  # threshold 파라미터 추가
   def decode_rle_to_mask(rle, height, width):
       s = rle.split()
       starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
       starts -= 1
       ends = starts + lengths
       img = np.zeros(height * width, dtype=np.uint8)
       
       for lo, hi in zip(starts, ends):
           img[lo:hi] = 1
       
       return img.reshape(height, width)

   def encode_mask_to_rle(mask):
       pixels = mask.flatten()
       pixels = np.concatenate([[0], pixels, [0]])
       runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
       runs[1::2] -= runs[::2]
       return ' '.join(str(x) for x in runs)

   # csv의 기본 column(column이지만 사실 row입니다.. default 8352)
   csv_column = 8352

   csv_data = []
   for path in csv_paths:
       data = pd.read_csv(path)
       csv_data.append(data)

   file_num = len(csv_data)
   filename_and_class = []
   rles = []

   print(f"앙상블할 모델 수: {file_num}, threshold: {threshold}")  # 정보 출력 추가

   for index in tqdm(range(csv_column)):    
       model_rles = []
       for data in csv_data:
           # rle 적용 시 이미지 사이즈는 변경하시면 안됩니다. 기본 test 이미지의 사이즈 그대로 유지하세요!
           if(type(data.iloc[index]['rle']) == float):
               model_rles.append(np.zeros((2048,2048)))
               continue
           model_rles.append(decode_rle_to_mask(data.iloc[index]['rle'],2048,2048))
       
       image = np.zeros((2048,2048))

       for model in model_rles:
           image += model
       
       # threshold 값으로 결정 (threshold의 값은 투표 수입니다!)
       # threshold로 설정된 값보다 크면 1, 작으면 0으로 변경합니다.
       image[image <= threshold] = 0
       image[image > threshold] = 1

       result_image = image

       rles.append(encode_mask_to_rle(result_image))
       filename_and_class.append(f"{csv_data[0].iloc[index]['class']}_{csv_data[0].iloc[index]['image_name']}")

   classes, filename = zip(*[x.split("_") for x in filename_and_class])
   image_name = [os.path.basename(f) for f in filename]

   # 기본 Dataframe의 구조는 image_name, class, rle로 되어있습니다.
   df = pd.DataFrame({
       "image_name": image_name,
       "class": classes,
       "rle": rles,
   })

   # 최종 ensemble output 저장
   df.to_csv(save_dir, index=False)
   

if __name__ == "__main__":
    csv_paths = ['후보 1 경로',
             '후보 2 경로',
             '후보 3 경로']

# threshold in [2]에서 []의 숫자를 변경하면 투표 임계값을 설정하실 수 있습니다.
for threshold in [2]:
    save_path = f"ensemble_threshold_{threshold}.csv"
    csv_ensemble(csv_paths, save_path, threshold)