import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse


def download_file(url: str, dst_path: str):
    """
    간단한 HTTP(S) 다운로드 (chunk 스트리밍)
    - 한글 주석: gdrive 링크는 gdown 설치 시 gdown으로 처리 시도
    """
    try:
        if "drive.google.com" in url or url.startswith("gdrive://"):
            try:
                import gdown  # type: ignore

                gdown.download(url, dst_path, quiet=False, fuzzy=True)
                return
            except Exception as _:
                pass
        import requests  # type: ignore

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"다운로드 실패: {e}")
        sys.exit(1)


def extract_archive(archive_path: str, target_dir: str):
    """
    zip/tar.gz 자동 해제
    """
    os.makedirs(target_dir, exist_ok=True)
    lower = archive_path.lower()
    try:
        if lower.endswith('.zip'):
            import zipfile

            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(target_dir)
        elif lower.endswith('.tar.gz') or lower.endswith('.tgz'):
            import tarfile

            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(target_dir)
        else:
            # 확장자 미확인: 폴더처럼 취급하여 target_dir로 이동만
            base = os.path.basename(archive_path)
            shutil.move(archive_path, os.path.join(target_dir, base))
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        sys.exit(1)


def call_core(args_list):
    cmd = [sys.executable, *args_list]
    print("RUN:", " ".join(map(str, cmd)))
    res = subprocess.run(list(map(str, cmd)))
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser(description="자동 다운로드→전처리→추출→학습")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_url", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", default="0")
    p.add_argument("--index_algorithm", default="Auto")
    p.add_argument("--embedder_model", default="contentvec")
    p.add_argument("--f0_method", default="rmvpe")
    args = p.parse_args()

    root_logs = os.path.join("logs", args.model_name)
    dst_dataset = os.path.join("assets", "datasets", args.model_name)
    os.makedirs(dst_dataset, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        parsed = urlparse(args.dataset_url)
        fname = os.path.basename(parsed.path) or "dataset.zip"
        archive_path = os.path.join(td, fname)
        print(f"다운로드: {args.dataset_url} -> {archive_path}")
        download_file(args.dataset_url, archive_path)
        print(f"압축 해제: {archive_path} -> {dst_dataset}")
        extract_archive(archive_path, dst_dataset)

    # 전처리(있을 때만 실행되도록 v2_core/core.py에서 처리)
    call_core(["rvc/v2_core/core.py", "preprocess", "--model_name", args.model_name, "--dataset_path", dst_dataset, "--sample_rate", 48000, "--cpu_cores", 4])

    # 특징 추출(있을 때만 실행)
    call_core([
        "rvc/v2_core/core.py",
        "extract",
        "--model_name",
        args.model_name,
        "--f0_method",
        args.f0_method,
        "--cpu_cores",
        4,
        "--gpu",
        args.gpu,
        "--sample_rate",
        48000,
        "--embedder_model",
        args.embedder_model,
        "--include_mutes",
        2,
    ])

    # 학습 + 인덱스
    call_core([
        "rvc/v2_core/core.py",
        "train",
        "--model_name",
        args.model_name,
        "--batch_size",
        args.batch_size,
        "--gpu",
        args.gpu,
        "--index_algorithm",
        args.index_algorithm,
    ])


if __name__ == "__main__":
    main()


