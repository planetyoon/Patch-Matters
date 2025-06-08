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
```
```bash
conda env create -f environment.yaml
conda activate patch_matters_divide
```
```bash
# 현재 가상환경 yaml 로 저장 방법
conda env create -f environment.yaml
```
```bash
# 공유받은 환경으로 그대로 환경 생성 방법
conda env create -f environment.yaml
conda activate patch_matters_divide
```


---

## ⚙️ 2. PyTorch 및 CUDA 호환 버전 설치

```bash
# 본인의 GPU 드라이버와 호환되는 CUDA 버전 확인 필요
# 예시: CUDA 11.6을 사용할 경우 (GPU 없는 경우에도 설치는 필요함)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
```

> ❗ **주의**: 로컬에서 GPU가 없을 경우 CUDA 디바이스 오류 발생함. 학습/추론은 서버에서 진행 권장.
---

## 📂 3. 필수 라이브러리 설치

```bash
# mmcv는 아래와 같이 CUDA, torch 버전에 맞게 설치
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmdet==3.1.0
pip install mmengine==0.10.1

#설치 확인
python -c "from mmdet.apis import DetInferencer; print('✅ DetInferencer import OK')"


# 기타 필수 라이브러리
pip install opencv-python-headless rich pillow tqdm
pip install ftfy
```



---

## 📁 4. 레포 구성

```
patchmatters-vessl/
├── divide/
│   ├── configs/
│   ├── checkpoints/  ← ✅ 학습된 모델 weight 저장
│   ├── data/         ← ✅ 결과 json 저장 폴더 (없으면 생성 필요)
│   ├── ovdet/        ← ✅ custom 모델 정의
│   ├── sample_tools/
│   ├── true_box_sample.py
│   └── generate_four_box.py
├── coco_image/
│   └── coco_sample_data_Image_Textualization/ ← ✅ 추론할 이미지 위치
```

> **필수 파일**:
>
> * `divide/ovdet/checkpoints/iter_90000.pth`
> * `divide/ovdet/data/metadata/coco_clip_hand_craft_attn12.npy`
---

## ✏️ 5. generate\_four\_box.py 코드 수정 사항

### 1. argparse → `arg` 대신 `args`로 전면 수정:

```python
arg = argparse.ArgumentParser()
...
args = arg.parse_args()
```

**수정 항목들:**

* `arg.image_folder → args.image_folder`
* `arg.four_box_save_path → args.four_box_save_path`
* `arg.object_box_save_path → args.object_box_save_path`
* 기타 `arg.`로 되어있던 모든 변수 → `args.`로 교체

### 2. GPU가 없는 경우 에러 방지용 디바이스 명시

```python
inference = DetInferencer(model=args.model_config_file, weights=args.checkpoint_file, device='cpu')
```

> ⚠️ `argparse`에서 `--device`를 인자로 받지 않기 때문에, 디바이스는 코드 내에서 직접 `device='cpu'`로 명시해야 함
---

## ▶️ 6. 실행 명령어 (로컬 or 서버)

```bash
# 수정됨
python generate_four_box.py \
  --model_config_file /root/Patch-Matters/divide/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
  --checkpoint_file /root/Patch-Matters/divide/ovdet/checkpoints/iter_90000.pth \
  --image_folder /root/Patch-Matters/coco_image/coco_sample_data_Image_Textualization \
  --four_box_save_path /root/Patch-Matters/divide/data/four_box.json \
  --object_box_save_path /root/Patch-Matters/divide/data/object_box.json

```

```bash
python ovdet/get_main_box.py ^
  --image_folder "C:/patchmatters-vessl/coco_image/coco_sample_data_Image_Textualization" ^
  --object_box_save_path "C:/patchmatters-vessl/divide/data/object_box.json" ^
  --main_box_save_path "C:/patchmatters-vessl/divide/data/main_box.json"
```


---

## ⚠️ 7. 주요 시행착오 정리

### ✅ 체크포인트 파일 경로 오류

* `FileNotFoundError` 발생 시 경로 확인 필수
* 파일명 오타: `this_repo_R-50-FPN_CLIP_iter_90000.pth` vs `iter_90000.pth`

### ✅ config와 weight 간 불일치 오류

* config 파일은 반드시 `ovdet`과 맞춰야 함 (`baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py`)

### ✅ custom registry 관련 warning

```
WARNING - Failed to search registry with scope "mmdet" in the "baron" registry tree.
```

* 이는 `ovdet` 내부에 정의된 custom registry 사용에 따른 것으로, 실제 inference 결과에 영향 없음

### ✅ inference 속도 매우 느림 (로컬)

* GPU 미탑재 시스템에서는 인퍼런스가 극도로 느림 → 서버에서 실행 권장

### ✅ CUDA 오류

```
RuntimeError: Found no NVIDIA driver on your system.
```

* 로컬에서 CUDA 디바이스를 사용하려 할 경우 발생. `device='cpu'`로 명시하거나 서버에서 실행 필요

### ✅ argparse 오타

```python
arg.image_folder → ❌
args.image_folder → ✅
```

### ✅ `.json` 저장이 안될 때

* `data/` 폴더 미생성 → 수동 생성 필요
* 터미널에서 파일 생성 여부 확인:

conda env create -f patch_matters_divide_env.yam용
```bash
watch -n 2 ls -lh divide/data
```

---

## ✅ 성공 조건 체크

* 터미널 출력에 `image path is doing:`이 반복적으로 출력됨
* `four_box.json`, `object_box.json` 크기가 점점 커짐

---


---


# Patch Matters Divide 원격 서버 실행 가이드

## ✅ 1⃣ 서버 접속

ssh [서버 아이디]@[서버 주소]

## ✅ 2⃣ 필수 환경 준비


### 서버에 conda 설치 /home/opt 에 설치 권장(innode 제한 없음...)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
# conda 최신 버전 설치
conda update -n base -c defaults conda 
```
### Conda 환경 생성
```bash

conda env create -f environment.yaml
conda activate patch_matters_divide
conda env create -f patch_matters_divide_env.yaml
#conda create -n patch_matters_divide python=3.8.19 -y
pip install -r requirements_divide.txt 


#별도 설치 필요
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
#설치 확인
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```


### conda 설치 했는데 conda --version 안나올떄
✅ 1️⃣ conda가 설치된 bin 경로 확인
지금 /root/miniconda3/bin/conda 가 실제로 있음.

확인:

```bash
ls /root/miniconda3/bin/conda
```
✅ 2️⃣ PATH 에 수동으로 추가

임시로 확인용:

```bash
export PATH=/root/miniconda3/bin:$PATH
```
그 다음:

```bash
conda --version
```
→ 이러면 아마 conda 23.x.x 이런 식으로 나오게 될 거예요.

### Git clone
```bash
git clone https://github.com/planetyoon/Patch-Matters.git
cd Patch-Matters/divide
```



---

# 🚀 get_main_box.py 실행 가이드 (최신/정리본)
## 📂 1️⃣ 사전 준비
### ✅ 먼저 generate_four_box.py 실행 완료 후 아래 두 파일이 반드시 존재해야 함:

파일	경로 (예시)
object_box.json	/root/Patch-Matters/divide/data/object_box.json
이미지 폴더 (your image folder)	/root/Patch-Matters/coco_image/coco_sample_data_Image_Textualization

→ 확인 방법:

```bash
ls -lh /root/Patch-Matters/divide/data/object_box.json
ls -lh /root/Patch-Matters/coco_image/coco_sample_data_Image_Textualization
```

---

## ✏️ 2️⃣ get_main_box.py 코드 수정 사항
### ✅ 수정 1️⃣ argparse 수정
기존:
```bash
arg.add_argument(...)
```
변경:
```bash
args = parser.parse_args()
args.llm_path = ...
```
→ 즉, 모든 arg. → args. 로 변경 필요

---

### ✅ 수정 2️⃣ LLM path 수정
```bash
arg.add_argument('--llm_path', type=str, help='LLM model', default='meta-llama/Meta-Llama-3-8B-Instruct')
```

---


### ⚠️ 3️⃣ 추가 패키지 설치 (가상환경 내 실행)
```bash
pip install --upgrade transformers
pip install icecream
pip install 'accelerate>=0.26.0'
pip install --upgrade jinja2
huggingface-cli login
```
##### huggingface-cli login 시 주의:

로그인 시 토큰 입력 → 토큰 발급은 👉 HuggingFace Tokens 에서 "New token" 생성

권한: 최소 read 권한 이상

터미널에서 로그인 성공 여부 확인:

```bash
huggingface-cli whoami
```

---


### ✅ 4️⃣ get_main_box.py 코드 추가 수정
#### 수정 1️⃣ generate_description 함수 수정
```bash
def generate_description(image_path, model, processor):
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    prompt = "Describe this image in one sentence."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text
```
#### 수정 2️⃣ re_match 함수 수정
```bash
def re_match(common_objects):
    import re
    pattern = r"\[(.*?)\]"
    matched_content = re.findall(pattern, str(common_objects))
    if not matched_content:
        return []
    cleaned_items = matched_content[0].replace("'", "").strip()
    return [item.strip() for item in cleaned_items.split(",") if item.strip()]
```

---

### ▶️ 5️⃣ 실행 명령어
```bash
python /root/Patch-Matters/divide/get_main_box.py \
  --image_folder /root/Patch-Matters/coco_image/coco_sample_data_Image_Textualization \
  --object_box_save_path /root/Patch-Matters/divide/data/object_box.json \
  --main_box_save_path /root/Patch-Matters/divide/data/main_box.json
```
#### 📌 결과 확인
실행 후:
```bash
ls -lh /root/Patch-Matters/divide/data/main_box.json
```
→ main_box.json 생성 확인

→ 이후 단계에서 description_generate/run.sh 에서 main_box.json 사용 예정

### 🚩 최종 주의사항
✅ generate_four_box.py → 반드시 먼저 실행 완료 후 진행
✅ get_main_box.py 는 수정사항 반영 후 실행 (안 그러면 오류 발생)
✅ main_box.json 은 data 폴더 아래 정상 생성됨 확인

---
