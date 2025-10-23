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
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


app = FastAPI(title="RVC V2 Trainer API", version="1.0.0")

# 한글 주석: 로컬 개발/트리거 편의 CORS 허용(필요시 제한하세요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        if not v and not values.get("prefix"):
            raise ValueError("'key' 또는 'prefix' 중 하나는 반드시 지정해야 합니다.")
        return v


class TrainRequest(BaseModel):
    """
    한글 주석: 학습 트리거 페이로드 (JSON)
    """

    model_name: str
    s3: S3Spec
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
        req.batch_size,
        "--gpu",
        req.gpu,
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
        # 1) S3에서 데이터 수신
        dst_dataset = os.path.join("assets", "datasets", req.model_name)
        _ensure_dir(dst_dataset)
        s3 = _make_s3_client(req.s3)

        if req.s3.key:
            with tempfile.TemporaryDirectory() as td:
                fname = os.path.basename(req.s3.key) or "dataset.zip"
                archive_path = os.path.join(td, fname)
                _download_s3_key(s3, req.s3.bucket, req.s3.key, archive_path)
                _extract_archive(archive_path, dst_dataset)
        else:
            _sync_s3_prefix(s3, req.s3.bucket, req.s3.prefix or "", dst_dataset)

        # 2) 파이프라인 실행
        _run_pipeline_with_local_dataset(req, logs_dir)

        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


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
    JOBS[job_id] = {
        "status": "queued",
        "logs_dir": os.path.join("logs", payload.model_name),
        "error": None,
        "request": json.loads(payload.json()),
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


