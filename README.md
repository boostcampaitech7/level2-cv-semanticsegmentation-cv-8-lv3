
#  🌏 Project Abstract
![image](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/92123779-aeca-4f97-bbda-ea5591ab9860.png)


뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

1. 질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.
2. 수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.
3. 의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.
4. 의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

<br />


<br />

## 🧑🏻‍🚀 Team Members

<div align="center">
<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/2113ee6a-f195-46c6-984d-00695bd87e97" width="140" height="140"><br/><a href="https://github.com/SeoJinHyoung" target="_blank">서진형</a></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/7f3b0715-5fb5-4292-ae93-322eb945898c" width="140" height="140"><br/><a href="https://github.com/kimgeonsu" target="_blank">김건수</a></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/fb1859a7-454d-44d1-97bb-4a752a57832d" width="140" height="140"><br/><a href="https://github.com/sihari-1115" target="_blank">이시하</a></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/2773af48-baa3-4fae-86e4-e897ed90a7e4" width="140" height="140"><br/><a href="https://github.com/One-HyeWon" target="_blank">조혜원</a></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/44d4d5e0-676e-4d01-9308-32e8fd9b2a13" width="140" height="140"><br/><a href="https://github.com/ruka030809" target="_blank">김형준</a></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/2113ee6a-f195-46c6-984d-00695bd87e97" width="140" height="140"><br/><a href="https://github.com/alexminyoungpark" target="_blank">박민영</a></td>
    </tr>
</table>
</div>

<br />

## 🗓️ Project Schedule
  2024/11/13 ~ 2024/11/28
  
<br />

## 🚀 Commit Convention

| Type       | Description            |
|------------|------------------------|
| `Feat`  | **새로운 기능을 추가**      |
| `Fix`      | **버그 수정**            |
| `Refactor` | **코드 리팩토링**         |
| `Experiment` | **실험용 코드**          |
| `Comment`  | **주석 추가 및 수정**     |
| `Remove`  | **파일을 삭제하는 작업만 수행한 경우**     |

#### Example
```shell
git commit -m "[#issue] Feature : message content"
```

<br />

## 📂 Directory Structure

```
├── 📄 .github          
├── 📄 .gitmodules
├── 📁 EDA/               # 데이터 탐색 코드
├── 📁 Ultralytics/       # YOLO 모델 관련 코드
├── 📁 base/              # 기본 설정 및 메인 코드
├── 📁 mmseg_base/        # MMSegmentation 관련 설정
├── 📁 utils/             # 유틸리티 코드 및 도구 ( 시각화, 앙상블, K-fold, GradCAM 등 )
└── 📄 README.md          # 프로젝트 개요 문서
```

<br />

## 🏆 LeaderBoard Results

### Public - Rank 19th **dice : 0.9725**

*1st. dice : 0.9759*

<img width="608" alt="image" src="https://github.com/user-attachments/assets/d7276589-bea6-4d45-a32d-11640bb486d2">

### Private - 19th **dice : 0.9738**

*1st. dice : 0.9771*

<img width="595" alt="image" src="https://github.com/user-attachments/assets/425f4364-39aa-4923-ab78-de4686114cd0">

<br />

<br />

## 📊 Final Performance Evaluation

| Architecture       | Encoder | LB Score |
|------------|---------------|--------------|
| UNet | EfficientNet-b7        | 0.9734         |
| UNet++ | HRNet_w64        | 0.9738         |
| UNet | ResNet50        | 0.9738         |

**UNet**은 대칭적인 U자 형태의 구조로, Encoder-Decoder 간의 Skip Connection을 통해 고해상도의 세밀한 특징을 효과적으로 보존할 수 있어 뼈와 같은 정밀한 구조를 추출하는 데 적합해서 선정되었습니다.
- **EfficientNet-B7**: 높은 파라미터 효율성과 강력한 특징 추출 능력으로, UNet 구조와 결합 시 연산 효율성과 정확도를 동시에 향상시켜주었습니다.
- **HRNet_W64**: 다양한 해상도에서의 정보를 유지하며 복합적인 세부 사항을 추출할 수 있어 UNet++과 함께 세밀한 특징을 효과적으로 학습합니다.
- **ResNet50**: 깊은 Residual 구조로 강력하고 안정적인 학습을 가능하게 하며, UNet과 결합해 균형 잡힌 성능을 제공합니다.

<br />

## 🤖 Ensemble

### Hard Voting

| Target       | LB Score       |
|-------------|-------------|
| ResNet + Efficient + HRNet     | 0.9735     |
| ResNet + Efficient     | 0.9738     |
| YOLO + Efficient(Soft Ensembled)     | 0.9689     |



### Soft Voting

| Target       | LB Score       |
|-------------|-------------|
| Efficient Fold 1,2,3,4,5     | 0.9734     |
| HRNet Fold 1,2,3,4,5     | 0.9738     |

