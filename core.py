import os
import sys
import argparse
import subprocess
import tempfile
import tarfile
import glob

import boto3  # S3 접근 (IAM/ENV 사용)

python = sys.executable


def run_preprocess(model_name: str, dataset_path: str, sample_rate: int, cpu_cores: int):
    """
    데이터 전처리. 원본 스크립트(rvc/train/preprocess/preprocess.py)에 의존
    - 한글 주석: v2_core만 업로드 시 이 단계는 건너뛰어야 함
    """
    rvc_root = os.environ.get("RVC_ROOT", "rvc")
    preprocess_script_path = os.path.join(rvc_root, "train", "preprocess", "preprocess.py")
    if not os.path.exists(preprocess_script_path):
        print("preprocess.py 가 존재하지 않아 이 단계를 건너뜁니다. (v2_core만 업로드한 경우)")
        return
    logs_path = os.path.join("logs", model_name)
    cmd = [
        python,
        preprocess_script_path,
        logs_path,
        dataset_path,
        str(sample_rate),
        str(cpu_cores),
        "Automatic",
        "False",
        "False",
        "0.7",
        "3.0",
        "0.3",
        "none",
    ]
    subprocess.run(list(map(str, cmd)))


def run_extract(model_name: str, f0_method: str, cpu_cores: int, gpu: str, sample_rate: int, embedder_model: str, include_mutes: int):
    """
    특징 추출. 원본 스크립트(rvc/train/extract/extract.py)에 의존
    - 한글 주석: v2_core만 업로드 시 이 단계는 건너뜁니다.
    """
    rvc_root = os.environ.get("RVC_ROOT", "rvc")
    extract_script = os.path.join(rvc_root, "train", "extract", "extract.py")
    if not os.path.exists(extract_script):
        print("extract.py 가 존재하지 않아 이 단계를 건너뜁니다. (v2_core만 업로드한 경우)")
        return
    logs_path = os.path.join("logs", model_name)
    cmd = [
        python,
        extract_script,
        logs_path,
        f0_method,
        str(cpu_cores),
        gpu,
        str(sample_rate),
        embedder_model,
        None,
        str(include_mutes),
    ]
    subprocess.run(list(map(str, cmd)))


def run_train(model_name: str, batch_size: int, gpu: str, index_algorithm: str):
    """
    학습 실행: v2_core 트레이너 사용, 48kHz/에폭 500/중간저장 없음 강제
    """
    trainer = os.path.join("rvc", "v2_core", "trainer.py")
    cmd = [
        python,
        trainer,
        model_name,
        500,          # save_every_epoch
        500,          # total_epoch
        "",          # pretrainG (자동 선택)
        "",          # pretrainD (자동 선택)
        gpu,
        batch_size,
        48000,        # sample_rate
        True,         # save_only_latest
        False,        # save_every_weights
        False,        # cache_data_in_gpu
        False,        # overtraining_detector
        50,           # overtraining_threshold
        False,        # cleanup
        "HiFi-GAN",  # vocoder
        False,        # checkpointing
    ]
    subprocess.run(list(map(str, cmd)))

    # 인덱스 생성
    indexer = os.path.join("rvc", "v2_core", "index.py")
    cmd_index = [python, indexer, os.path.join("logs", model_name), index_algorithm]
    subprocess.run(list(map(str, cmd_index)))


def s3_client(region: str | None = None, endpoint_url: str | None = None):
    """
    한글 주석: Pod의 IAM/ENV를 사용하여 S3 클라이언트 생성
    - 액세스 키는 API로 주고받지 않음
    """
    kwargs: dict[str, str] = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def s3_download_object(bucket: str, key: str, dst_path: str, region: str | None = None, endpoint_url: str | None = None):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    s3 = s3_client(region, endpoint_url)
    s3.download_file(bucket, key, dst_path)


def s3_upload_file(bucket: str, key: str, src_path: str, region: str | None = None, endpoint_url: str | None = None):
    s3 = s3_client(region, endpoint_url)
    s3.upload_file(src_path, bucket, key)


def s3_upload_artifacts(model_name: str, logs_dir: str, bucket: str, prefix: str, region: str | None = None, endpoint_url: str | None = None):
    """
    한글 주석: 학습 산출물(pth, index, config, 전체 로그 아카이브) 업로드
    """
    files: list[str] = []
    files += glob.glob(os.path.join(logs_dir, f"{model_name}_*e_*s.pth"))
    files += glob.glob(os.path.join(logs_dir, f"{model_name}*.index"))
    cfg = os.path.join(logs_dir, "config.json")
    if os.path.exists(cfg):
        files.append(cfg)
    tjson = os.path.join(logs_dir, "training_data.json")
    if os.path.exists(tjson):
        files.append(tjson)

    # logs 전체 아카이브
    archive_path = os.path.join(logs_dir, f"{model_name}_logs.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(logs_dir, arcname=os.path.basename(logs_dir))
    files.append(archive_path)

    for f in files:
        key = os.path.join(prefix, os.path.basename(f))
        s3_upload_file(bucket, key, f, region, endpoint_url)


def run_pipeline_uid(
    model_name: str,
    uid: str,
    bucket: str,
    region: str | None,
    endpoint_url: str | None,
    batch_size: int,
    gpu: str,
    index_algorithm: str,
):
    """
    한글 주석: UID 기반 S3 파이프라인
    - 입력: s3://{bucket}/vtubervoice/voice_blend/{uid}/uploads/{uid}.mp3
    - 출력: s3://{bucket}/vtubervoice/voice_blend/{uid}/results/*
    - 자격증명은 Pod의 IAM/ENV 사용
    """
    # 한글 주석: 유연한 프리픽스 처리
    # - S3_DATA_PREFIX: 예) "voice_blend/uploads/{uid}.mp3" 또는 "voice_blend/uploads/"
    # - S3_MODELS_PREFIX: 예) "voice_blend/models/{uid}" 또는 "voice_blend/models/"
    def _join(a: str, b: str) -> str:
        return a.rstrip("/") + "/" + b.lstrip("/")

    data_pref = os.environ.get("S3_DATA_PREFIX")
    models_pref = os.environ.get("S3_MODELS_PREFIX")
    base_prefix = os.environ.get("S3_BASE_PREFIX", "vtubervoice/voice_blend")

    if data_pref:
        if "{uid}" in data_pref:
            # 템플릿 전체가 파일 경로일 수 있음
            cand = data_pref.format(uid=uid)
            if cand.endswith(".mp3"):
                src_key = cand
            else:
                src_key = _join(cand, f"{uid}.mp3")
        else:
            src_key = _join(data_pref, f"{uid}.mp3")
    else:
        src_key = os.path.join(base_prefix, uid, "uploads", f"{uid}.mp3")

    if models_pref:
        if "{uid}" in models_pref:
            dst_prefix = models_pref.format(uid=uid)
        else:
            dst_prefix = _join(models_pref, uid)
    else:
        dst_prefix = os.path.join(base_prefix, uid, "results")

    # 1) S3에서 mp3 수신 → 데이터셋 디렉토리
    dataset_dir = os.path.join("assets", "datasets", model_name)
    os.makedirs(dataset_dir, exist_ok=True)
    local_mp3 = os.path.join(dataset_dir, f"{uid}.mp3")
    s3_download_object(bucket, src_key, local_mp3, region, endpoint_url)

    # 2) 전처리/특징추출(존재하면 실행, 없으면 core.py가 건너뜀)
    run_preprocess(model_name, dataset_dir, 48000, os.cpu_count() or 4)
    run_extract(model_name, "rmvpe", os.cpu_count() or 4, gpu, 48000, "contentvec", 2)

    # 3) 학습(+인덱스)
    run_train(model_name, batch_size, gpu, index_algorithm)

    # 4) 산출물 업로드
    logs_dir = os.path.join("logs", model_name)
    s3_upload_artifacts(model_name, logs_dir, bucket, dst_prefix, region, endpoint_url)


def parse_args():
    p = argparse.ArgumentParser(description="v2_core minimal CLI")
    sub = p.add_subparsers(dest="mode")

    sp = sub.add_parser("preprocess")
    sp.add_argument("--model_name", required=True)
    sp.add_argument("--dataset_path", required=True)
    sp.add_argument("--sample_rate", type=int, default=48000)
    sp.add_argument("--cpu_cores", type=int, default=4)

    se = sub.add_parser("extract")
    se.add_argument("--model_name", required=True)
    se.add_argument("--f0_method", default="rmvpe")
    se.add_argument("--cpu_cores", type=int, default=4)
    se.add_argument("--gpu", default="0")
    se.add_argument("--sample_rate", type=int, default=48000)
    se.add_argument("--embedder_model", default="contentvec")
    se.add_argument("--include_mutes", type=int, default=2)

    st = sub.add_parser("train")
    st.add_argument("--model_name", required=True)
    st.add_argument("--batch_size", type=int, default=8)
    st.add_argument("--gpu", default="0")
    st.add_argument("--index_algorithm", default="Auto")

    si = sub.add_parser("index")
    si.add_argument("--model_name", required=True)
    si.add_argument("--index_algorithm", default="Auto")

    spipe = sub.add_parser("pipeline")
    spipe.add_argument("--model_name", required=False, default=None)
    spipe.add_argument("--uid", required=True)
    spipe.add_argument("--bucket", required=False, default=os.environ.get("S3_BUCKET", ""))
    spipe.add_argument("--region", required=False, default=os.environ.get("S3_REGION", None))
    spipe.add_argument("--endpoint_url", required=False, default=os.environ.get("S3_ENDPOINT_URL", None))
    spipe.add_argument("--batch_size", type=int, default=8)
    spipe.add_argument("--gpu", default="0")
    spipe.add_argument("--index_algorithm", default="Auto")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "preprocess":
        run_preprocess(args.model_name, args.dataset_path, args.sample_rate, args.cpu_cores)
    elif args.mode == "extract":
        run_extract(args.model_name, args.f0_method, args.cpu_cores, args.gpu, args.sample_rate, args.embedder_model, args.include_mutes)
    elif args.mode == "train":
        run_train(args.model_name, args.batch_size, args.gpu, args.index_algorithm)
    elif args.mode == "index":
        indexer = os.path.join("rvc", "v2_core", "index.py")
        cmd_index = [python, indexer, os.path.join("logs", args.model_name), args.index_algorithm]
        subprocess.run(list(map(str, cmd_index)))
    elif args.mode == "pipeline":
        bucket = args.bucket or os.environ.get("S3_BUCKET")
        if not bucket:
            print("S3 버킷을 찾을 수 없습니다. --bucket 또는 S3_BUCKET 환경변수를 지정하세요.")
            sys.exit(1)
        # 한글 주석: model_name이 없으면 uid를 사용
        resolved_model_name = args.model_name or args.uid
        run_pipeline_uid(
            model_name=resolved_model_name,
            uid=args.uid,
            bucket=bucket,
            region=args.region,
            endpoint_url=args.endpoint_url,
            batch_size=args.batch_size,
            gpu=args.gpu,
            index_algorithm=args.index_algorithm,
        )
    else:
        print("사용법: preprocess | extract | train | index | pipeline 서브커맨드 중 하나를 선택하세요.")


if __name__ == "__main__":
    main()


