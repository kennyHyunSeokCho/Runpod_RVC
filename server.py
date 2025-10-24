import os
import sys
import uuid
import json
import shutil
import zipfile
import tarfile
import tempfile
import subprocess
from typing import Optional

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


load_dotenv()  # 한글 주석: .env 로드(S3_BUCKET, S3_REGION 등)

app = FastAPI(title="RVC V2 Trainer API", version="1.0.0")

# 한글 주석: 로컬 개발/트리거 편의 CORS 허용(필요시 제한하세요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 한글 주석: GPU 및 배치 사이즈 고정값(ENV로 오버라이드 가능)
DEFAULT_GPU = os.getenv("DEFAULT_GPU", "0")
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "8"))


class S3Spec(BaseModel):
    """
    한글 주석: S3에서 입력을 가져오는 명세
    - key: 단일 객체(보통 zip/tar)
    - prefix: 다수 객체(폴더 동기화)
    둘 중 하나는 반드시 지정되어야 함
    """

    bucket: str = Field(..., description="S3 버킷 이름")
    key: Optional[str] = Field(None, description="단일 객체 키 (zip/tar 등)")
    prefix: Optional[str] = Field(None, description="폴더 프리픽스 (여러 객체 다운로드)")
    region: Optional[str] = Field(None, description="AWS 리전, 예: ap-northeast-2")
    endpoint_url: Optional[str] = Field(None, description="커스텀 S3 엔드포인트 URL")
    aws_access_key_id: Optional[str] = Field(None, description="명시적 자격증명(옵션)")
    aws_secret_access_key: Optional[str] = Field(None, description="명시적 자격증명(옵션)")
    aws_session_token: Optional[str] = Field(None, description="세션 토큰(옵션)")

    @validator("key", always=True)
    def _key_or_prefix(cls, v, values):
        # 한글 주석: s3가 사용될 때만 key/prefix 검증. TrainRequest에서 s3는 선택사항
        if not v and not values.get("prefix"):
            raise ValueError("'key' 또는 'prefix' 중 하나는 반드시 지정해야 합니다.")
        return v


class TrainRequest(BaseModel):
    """
    한글 주석: 학습 트리거 페이로드 (JSON)
    """

    model_name: Optional[str] = None
    # 한글 주석: uid/metadata_url/dataset_url 중 하나로 소스 지정 가능, s3는 선택
    uid: Optional[str] = None
    metadata_url: Optional[str] = None
    dataset_url: Optional[str] = None
    s3: Optional[S3Spec] = None
    # 한글 주석: 학습 후 산출물 업로드 위치(미지정 시 uid 기반 기본 프리픽스 생성)
    class UploadSpec(BaseModel):
        bucket: Optional[str] = None
        prefix: Optional[str] = None  # 예: vtubervoice/voice_blend/{uid}/results
        region: Optional[str] = None
        endpoint_url: Optional[str] = None
        aws_access_key_id: Optional[str] = None
        aws_secret_access_key: Optional[str] = None
        aws_session_token: Optional[str] = None

    upload: Optional[UploadSpec] = None
    batch_size: int = 8
    gpu: str = "0"
    cpu_cores: int = os.cpu_count() or 4
    index_algorithm: str = "Auto"
    embedder_model: str = "contentvec"
    f0_method: str = "rmvpe"
    sample_rate: int = 48000


JOBS: dict[str, dict] = {}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _extract_archive(archive_path: str, target_dir: str):
    """
    한글 주석: zip/tar.gz 자동 해제
    """
    lower = archive_path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(target_dir)
    else:
        # 알 수 없는 확장자는 파일만 이동
        shutil.move(archive_path, os.path.join(target_dir, os.path.basename(archive_path)))


def _make_s3_client(spec: S3Spec):
    """
    한글 주석: S3 클라이언트 생성 (명시 자격증명 또는 인스턴스 역할)
    """
    kwargs = {}
    if spec.region:
        kwargs["region_name"] = spec.region
    if spec.endpoint_url:
        kwargs["endpoint_url"] = spec.endpoint_url
    if spec.aws_access_key_id and spec.aws_secret_access_key:
        kwargs.update(
            {
                "aws_access_key_id": spec.aws_access_key_id,
                "aws_secret_access_key": spec.aws_secret_access_key,
            }
        )
        if spec.aws_session_token:
            kwargs["aws_session_token"] = spec.aws_session_token
    return boto3.client("s3", **kwargs)


def _download_http(url: str, dst_file: str):
    """
    한글 주석: HTTP(S) 또는 gdrive 다운로드(가능하면 gdown 사용)
    """
    if "drive.google.com" in url or url.startswith("gdrive://"):
        try:
            import gdown  # type: ignore

            gdown.download(url, dst_file, quiet=False, fuzzy=True)
            return
        except Exception:
            pass
    import requests  # type: ignore

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _upload_artifacts_to_s3(model_name: str, upload_spec: TrainRequest.UploadSpec, src_logs_dir: str):
    """
    한글 주석: 학습 산출물(.pth/.index 및 logs 아카이브)을 S3에 업로드
    """
    if not upload_spec or not (upload_spec.bucket and upload_spec.prefix):
        return
    # 클라이언트 생성
    spec = upload_spec
    kwargs = {}
    if spec.region:
        kwargs["region_name"] = spec.region
    if spec.endpoint_url:
        kwargs["endpoint_url"] = spec.endpoint_url
    if spec.aws_access_key_id and spec.aws_secret_access_key:
        kwargs.update(
            {
                "aws_access_key_id": spec.aws_access_key_id,
                "aws_secret_access_key": spec.aws_secret_access_key,
            }
        )
        if spec.aws_session_token:
            kwargs["aws_session_token"] = spec.aws_session_token
    s3 = boto3.client("s3", **kwargs)

    # 업로드 대상 수집
    import glob
    files = []
    files += glob.glob(os.path.join(src_logs_dir, f"{model_name}_*e_*s.pth"))
    files += glob.glob(os.path.join(src_logs_dir, f"{model_name}*.index"))
    files += [os.path.join(src_logs_dir, "config.json")]
    files += [os.path.join(src_logs_dir, "training_data.json")] if os.path.exists(os.path.join(src_logs_dir, "training_data.json")) else []

    # logs 아카이브 생성
    archive_path = os.path.join(src_logs_dir, f"{model_name}_logs.tar.gz")
    import tarfile as _tarfile
    with _tarfile.open(archive_path, "w:gz") as tf:
        tf.add(src_logs_dir, arcname=os.path.basename(src_logs_dir))
    files.append(archive_path)

    # 업로드
    for f in files:
        if not os.path.exists(f):
            continue
        key = os.path.join(spec.prefix, os.path.basename(f))
        s3.upload_file(f, spec.bucket, key)


def _download_s3_key(client, bucket: str, key: str, dst_file: str):
    client.download_file(bucket, key, dst_file)


def _sync_s3_prefix(client, bucket: str, prefix: str, dst_dir: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            rel = os.path.relpath(k, prefix)
            local_path = os.path.join(dst_dir, rel)
            _ensure_dir(os.path.dirname(local_path))
            client.download_file(bucket, k, local_path)


def _run_pipeline_with_local_dataset(req: TrainRequest, logs_dir: str):
    """
    한글 주석: main.py를 호출해 로컬 데이터셋 경로로 파이프라인 실행
    """
    cmd = [
        sys.executable,
        os.path.join(os.getcwd(), "main.py"),
        "--model_name",
        req.model_name,
        "--dataset_path",
        os.path.join("assets", "datasets", req.model_name),
        "--batch_size",
        DEFAULT_BATCH_SIZE,
        "--gpu",
        DEFAULT_GPU,
        "--index_algorithm",
        req.index_algorithm,
        "--embedder_model",
        req.embedder_model,
        "--f0_method",
        req.f0_method,
        "--sample_rate",
        req.sample_rate,
    ]

    # 한글 주석: 로그 파일로 표준 출력/에러 리다이렉트
    _ensure_dir(logs_dir)
    log_path = os.path.join(logs_dir, "server_job.log")
    with open(log_path, "ab", buffering=0) as lf:
        subprocess.run(list(map(str, cmd)), stdout=lf, stderr=lf)


def _background_worker(job_id: str, req: TrainRequest):
    logs_dir = os.path.join("logs", req.model_name)
    JOBS[job_id] = {"status": "running", "logs_dir": logs_dir, "error": None}
    try:
        # 1) 데이터 소스 해석 및 수신 (dataset_url | metadata_url+uid | s3)
        dst_dataset = os.path.join("assets", "datasets", req.model_name)
        _ensure_dir(dst_dataset)
        # (A) dataset_url이 있으면 HTTP로 단일 압축 파일 다운로드
        if req.dataset_url:
            with tempfile.TemporaryDirectory() as td:
                parsed_name = os.path.basename(req.dataset_url) or "dataset.zip"
                archive_path = os.path.join(td, parsed_name)
                _download_http(req.dataset_url, archive_path)
                _extract_archive(archive_path, dst_dataset)
        # (B) metadata_url이 있는 경우 메타 JSON을 읽고 s3 또는 url을 해석
        elif req.metadata_url:
            import json as _json
            with tempfile.TemporaryDirectory() as td:
                meta_path = os.path.join(td, "meta.json")
                _download_http(req.metadata_url, meta_path)
                with open(meta_path, "r") as f:
                    meta = _json.load(f)
                # 우선순위: dataset_url -> s3.key/prefix
                if meta.get("dataset_url"):
                    with tempfile.TemporaryDirectory() as t2:
                        archive_path = os.path.join(t2, os.path.basename(meta["dataset_url"]) or "dataset.zip")
                        _download_http(meta["dataset_url"], archive_path)
                        _extract_archive(archive_path, dst_dataset)
                elif meta.get("bucket") and (meta.get("key") or meta.get("prefix")):
                    s3_spec = S3Spec(
                        bucket=meta["bucket"],
                        key=meta.get("key"),
                        prefix=meta.get("prefix"),
                        region=meta.get("region"),
                        endpoint_url=meta.get("endpoint_url"),
                        aws_access_key_id=meta.get("aws_access_key_id"),
                        aws_secret_access_key=meta.get("aws_secret_access_key"),
                        aws_session_token=meta.get("aws_session_token"),
                    )
                    s3 = _make_s3_client(s3_spec)
                    if s3_spec.key:
                        with tempfile.TemporaryDirectory() as t2:
                            archive_path = os.path.join(t2, os.path.basename(s3_spec.key) or "dataset.zip")
                            _download_s3_key(s3, s3_spec.bucket, s3_spec.key, archive_path)
                            _extract_archive(archive_path, dst_dataset)
                    else:
                        _sync_s3_prefix(s3, s3_spec.bucket, s3_spec.prefix or "", dst_dataset)
                else:
                    raise RuntimeError("metadata_url에 dataset_url 또는 s3(bucket+key/prefix)가 없습니다.")
        # (C) s3가 직접 제공된 경우
        elif req.s3 is not None:
            s3 = _make_s3_client(req.s3)
            if req.s3.key:
                with tempfile.TemporaryDirectory() as td:
                    fname = os.path.basename(req.s3.key) or "dataset.zip"
                    archive_path = os.path.join(td, fname)
                    _download_s3_key(s3, req.s3.bucket, req.s3.key, archive_path)
                    _extract_archive(archive_path, dst_dataset)
            else:
                _sync_s3_prefix(s3, req.s3.bucket, req.s3.prefix or "", dst_dataset)
        # (D) uid만 있는 경우: ENV(DATASET_PREFIX) 기반 프리픽스 추정
        elif req.uid is not None:
            base_prefix = os.getenv("DATASET_PREFIX", "")
            inferred_prefix = f"{base_prefix}/{req.uid}" if base_prefix else req.uid
            if not req.s3 or not req.s3.bucket:
                raise RuntimeError("uid만 전달된 경우 s3.bucket 정보가 필요합니다.")
            s3 = _make_s3_client(req.s3)
            _sync_s3_prefix(s3, req.s3.bucket, inferred_prefix, dst_dataset)
        else:
            raise RuntimeError("데이터 소스를 찾을 수 없습니다. dataset_url | metadata_url | s3 | uid 중 하나를 제공하세요.")

        # 2) 파이프라인 실행
        _run_pipeline_with_local_dataset(req, logs_dir)

        # 3) 산출물 업로드 (업로드 명세가 없으면 uid 기반 기본값 구성)
        target_upload = req.upload
        if (not target_upload or not target_upload.prefix) and req.uid:
            default_prefix = f"vtubervoice/voice_blend/{req.uid}/results"
            if not target_upload:
                target_upload = TrainRequest.UploadSpec(bucket=(req.s3.bucket if req.s3 else None), prefix=default_prefix)
            else:
                if not target_upload.prefix:
                    target_upload.prefix = default_prefix
                if not target_upload.bucket and req.s3:
                    target_upload.bucket = req.s3.bucket
        logs_dir = os.path.join("logs", req.model_name)
        _upload_artifacts_to_s3(req.model_name, target_upload, logs_dir)

        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.get("/")
def root():
    return {"name": "RVC V2 Trainer API", "endpoints": ["GET /health", "POST /train", "GET /jobs/{id}"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def trigger_train(payload: TrainRequest, background_tasks: BackgroundTasks):
    """
    한글 주석: 학습 트리거
    - 요청 즉시 202 형태로 job_id 반환
    - 백그라운드에서 S3 다운로드→파이프라인 실행
    """
    job_id = uuid.uuid4().hex
    resolved_model = payload.model_name or payload.uid
    if not resolved_model:
        return {"error": "model_name_or_uid_required"}
    JOBS[job_id] = {
        "status": "queued",
        "logs_dir": os.path.join("logs", resolved_model),
        "error": None,
        "request": json.loads(payload.json()),
    }
    # 기존 로컬 파이프라인 호출 대신, uid+s3.bucket 이 오면 core.py pipeline을 직접 호출
    if payload.uid:
        env_bucket = os.getenv("S3_BUCKET", "")
        env_region = os.getenv("S3_REGION")
        env_endpoint = os.getenv("S3_ENDPOINT_URL")
        cmd = [
            sys.executable,
            os.path.join(os.getcwd(), "core.py"),
            "pipeline",
            "--model_name", resolved_model,
            "--uid", payload.uid,
            "--bucket", env_bucket if env_bucket else (payload.s3.bucket if payload.s3 else ""),
            "--gpu", DEFAULT_GPU,
            "--batch_size", str(DEFAULT_BATCH_SIZE),
            "--index_algorithm", payload.index_algorithm,
        ]
        if env_region:
            cmd += ["--region", env_region]
        if env_endpoint:
            cmd += ["--endpoint_url", env_endpoint]
        background_tasks.add_task(lambda: subprocess.run(list(map(str, cmd))))
        return {
            "job_id": job_id,
            "status": JOBS[job_id]["status"],
            "logs_dir": JOBS[job_id]["logs_dir"],
            "mode": "pipeline",
        }
    background_tasks.add_task(_background_worker, job_id, payload)
    return {
        "job_id": job_id,
        "status": JOBS[job_id]["status"],
        "logs_dir": JOBS[job_id]["logs_dir"],
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "not_found"}
    return job


# 한글 주석: uvicorn 직접 실행을 위한 진입점
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


