# v2_core 사용법 (Runpod/Colab 유사 환경)

## 필수 준비
- Python 3.10+ 권장
- GPU 환경(CUDA) 권장 (없어도 학습은 느리지만 동작)
- ffmpeg 설치 필요 (setup_runpod.sh가 자동 설치)

## 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Runpod(우분투)에서 한 번에:
```bash
bash rvc/v2_core/setup_runpod.sh
```

## 데이터 배치
- 입력: `assets/datasets/<모델명>` 폴더에 오디오 파일들 배치(또는 URL 사용)
- 출력: `logs/<모델명>`

## S3 기반 원샷 파이프라인 (UID)
- 환경변수(.env 또는 Pod ENV)
```
S3_BUCKET=your-bucket
S3_REGION=ap-southeast-2
# 입력 키 템플릿: {uid} 치환. 파일 키를 직접 지정하거나, 폴더만 주면 자동으로 {uid}.mp3를 붙입니다.
S3_DATA_PREFIX=voice_blend/uploads/{uid}.mp3
# 출력 프리픽스 템플릿: {uid} 치환
S3_MODELS_PREFIX=voice_blend/models/{uid}
DEFAULT_GPU=0
DEFAULT_BATCH_SIZE=8
```

- 실행(런포드 쉘)
```bash
python core.py pipeline --model_name <모델명> --uid <uid>
```
  - 동작: S3에서 입력 mp3 다운로드 → 전처리 → 특징추출 → 학습(+인덱스) → 결과물을 S3_MODELS_PREFIX로 업로드

## 원클릭 자동 학습 (URL)
```bash
python rvc/v2_core/auto_train.py \
  --model_name <모델명> \
  --dataset_url <zip_or_tar.gz_URL> \
  --batch_size 8 \
  --gpu 0
```

#api
curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/train" \
  -H "Content-Type: application/json" \
  -d '{"uid":"test"}'

## 수동 파이프라인 (로컬 데이터)
```bash
python rvc/v2_core/core.py preprocess --model_name <모델명> --dataset_path assets/datasets/<모델명> --sample_rate 48000
python rvc/v2_core/core.py extract --model_name <모델명> --f0_method rmvpe --gpu 0 --sample_rate 48000 --embedder_model contentvec
python rvc/v2_core/core.py train --model_name <모델명> --batch_size 8 --gpu 0 --index_algorithm Auto
```

## 고정 설정
- 샘플레이트: 48000Hz 고정
- Epoch: 500 고정
- 중간 체크포인트 저장: 없음 (최종만 저장)
- 무음 파일: 학습 시작 시 자동 생성/보장(`ensure_mute_assets`)

## 인덱스 파일
- 학습 종료 후 `logs/<모델명>/<모델명>.index` 자동 생성
- 내부 구현: FAISS IVF, 필요 시 MiniBatchKMeans로 샘플 축약
- CPU만 있을 경우 `faiss-cpu`로 동작, GPU FAISS는 환경에 따라 별도 설치 필요

## 사전 학습 모델
- `rvc/models`에 필요한 임베더(contentvec 등)와 F0 모델(rmvpe)이 있다면 사용
- 없으면 추후 다운로드/경로 지정 필요 (v2_core는 존재 시 사용)

## 참고
- FastAPI API 사용 시(포트 8000 노출):
  - 서버 기동: `uvicorn server:app --host 0.0.0.0 --port 8000`
  - 트리거 예시:
    ```bash
    curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/train" \
      -H "Content-Type: application/json" \
      -d '{
        "model_name": "my_model",
        "uid": "test"
      }'
    ```
  - 설명: 서버는 uid와 .env의 S3 설정을 사용해 `core.py pipeline`을 백그라운드로 실행합니다.

### Runpod 셋업/실행(처음부터)
```bash
bash -lc 'set -e
cd /workspace

# 1) 레포 클론
git clone https://github.com/kennyHyunSeokCho/Runpod_RVC.git
cd Runpod_RVC

# 2) 의존성 설치
python3 -m pip install --upgrade pip
pip install -r requirements.txt
# ffmpeg 설치가 필요하면(없을 때만)
bash setup_runpod.sh || true

# 3) Applio에서 전처리/추출 코드만 가져오기
git clone --depth 1 https://github.com/IAHispano/Applio.git /tmp/applio
mkdir -p rvc
rsync -av --delete /tmp/applio/rvc/train/ rvc/train/
rsync -av --delete /tmp/applio/rvc/lib/   rvc/lib/

# 4) .env 작성(값을 실제로 채워 넣으세요)
cat > .env <<EOF
S3_BUCKET=your-bucket
S3_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=REDACTED
AWS_SECRET_ACCESS_KEY=REDACTED
AWS_DEFAULT_REGION=ap-southeast-2
S3_DATA_PREFIX=voice_blend/uploads/{uid}.mp3
S3_MODELS_PREFIX=voice_blend/models/{uid}
DEFAULT_GPU=0
DEFAULT_BATCH_SIZE=8
EOF

# 5-1) 한 번에 실행(UID 기반)
python core.py pipeline --uid my_uid

# 5-2) API 서버로 실행(선택)
# uvicorn server:app --host 0.0.0.0 --port 8000
# curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/train" -H "Content-Type: application/json" -d "{\"uid\":\"my_uid\"}"
'
```
- Windows: `rvc/v2_core/run-v2-train.bat` 사용 가능
- 로그 경로: `logs/<모델명>`
