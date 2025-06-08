# Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception



## Usage

### Checkpoints

google drive에 올려놨습니다. 다운받으시고 아래 경로 확인 후에 경로대로 넣으시면 됩니다.
로컬에 다운받으시고 SSH 접속후 VSCODE 에서 Explorer 에서 Drag and Drop으로 넣는게 편함.

`ovdet/checkpoints`
- clip_vitb32.pth
- res50_fpn_soco_star_400.pth
- this_repo_R-50-FPN_CLIP_iter_90000.pth

`ovdet/data/coco/annotations`
- instances_train2017.json
- instances_val2017.json

`ovdet/data/coco/wusize` 
https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG 여기서 받으시면 됩니다.
- captions_train2017_tags_allcaps.json
- instances_train2017_base.json -> 위 구글드라이브 링크에서 받으면됨. 
- instances_train2017_novel.json
- instances_val2017_base.json
- instances_val2017_novel.json



# Patch Matters Divide 환경 구축 및 generate\_four\_box 실행 가이드 (w/ 시행착오 정리)

이 문서는 Patch Matters 프로젝트의 `divide` 환경을 처음부터 구축하고 `generate_four_box.py` 스크립트를 성공적으로 실행하기까지의 **모든 과정과 시행착오**를 기록한 문서입니다.

---

## 📦 1. Conda 환경 구성 패키지 파일 설치까지 한번에 포함해둠.
### 방법 1(빠른 방법)
```bash
conda create -n patch_matters_divide python=3.8.19 -y
conda activate patch_matters_divide
pip install -r requirements_divide.txt
```
### 방법 2(추천, 정확한 복원/env 최초 생성 시 및 서버에서 사용 권장) -> 윈도우 기반으로 되어 있어 리눅스 기반으로 변경중 
```bash
conda env create -f patch_matters_divide_env.yam용
```bash
conda env create -f environment.yaml
conda activate patch_matters_divide

```


