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

## 원클릭 자동 학습 (URL)
```bash
python rvc/v2_core/auto_train.py \
  --model_name <모델명> \
  --dataset_url <zip_or_tar.gz_URL> \
  --batch_size 8 \
  --gpu 0
```

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
- Windows: `rvc/v2_core/run-v2-train.bat` 사용 가능
- 로그 경로: `logs/<모델명>`
