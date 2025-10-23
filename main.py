import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse


def _run(cmd: list[str]):
    """
    한글 주석: 서브프로세스로 명령 실행하는 유틸리티
    에러 발생 시 동일 코드로 종료
    """
    print("$", " ".join(map(str, cmd)))
    ret = subprocess.run(list(map(str, cmd)))
    if ret.returncode != 0:
        sys.exit(ret.returncode)


def _resolve_core_path() -> str:
    """
    한글 주석: core.py 위치 자동 탐색
    - 우선순위: rvc/v2_core/core.py -> ./core.py
    """
    root = os.getcwd()
    nested = os.path.join(root, "rvc", "v2_core", "core.py")
    local = os.path.join(root, "core.py")
    if os.path.exists(nested):
        return nested
    if os.path.exists(local):
        return local
    print("core.py를 찾을 수 없습니다. rvc/v2_core 또는 현재 디렉토리에 배치해 주세요.")
    sys.exit(1)


def _download_file(url: str, dst_path: str):
    """
    한글 주석: 간단한 HTTP(S)/Google Drive 다운로드
    - gdrive는 gdown이 설치되어 있으면 gdown 사용 시도
    """
    try:
        if "drive.google.com" in url or url.startswith("gdrive://"):
            try:
                import gdown  # type: ignore

                gdown.download(url, dst_path, quiet=False, fuzzy=True)
                return
            except Exception:
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


def _extract_archive(archive_path: str, target_dir: str):
    """
    한글 주석: zip/tar.gz 자동 해제, 기타 확장자는 파일 이동 처리
    """
    os.makedirs(target_dir, exist_ok=True)
    lower = archive_path.lower()
    try:
        if lower.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
        elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            import tarfile

            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(target_dir)
        else:
            base = os.path.basename(archive_path)
            shutil.move(archive_path, os.path.join(target_dir, base))
    except Exception as e:
        print(f"압축 해제 실패: {e}")
        sys.exit(1)


def run_pipeline(
    model_name: str,
    dataset_path: str | None,
    dataset_url: str | None,
    batch_size: int,
    gpu: str,
    cpu_cores: int,
    index_algorithm: str,
    embedder_model: str,
    f0_method: str,
    sample_rate: int,
):
    """
    한글 주석: 전처리 -> 특징추출 -> 학습(+인덱스)까지 한 번에 실행
    - dataset_url이 있으면 다운로드/압축해제 후 진행
    - 없는 경우 dataset_path를 사용
    """
    core_py = _resolve_core_path()

    # 데이터셋 경로 결정 및 URL 다운로드
    if dataset_url:
        dst_dataset = os.path.join("assets", "datasets", model_name)
        os.makedirs(dst_dataset, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            parsed = urlparse(dataset_url)
            fname = os.path.basename(parsed.path) or "dataset.zip"
            archive_path = os.path.join(td, fname)
            print(f"다운로드: {dataset_url} -> {archive_path}")
            _download_file(dataset_url, archive_path)
            print(f"압축 해제: {archive_path} -> {dst_dataset}")
            _extract_archive(archive_path, dst_dataset)
        dataset_path = dst_dataset
    else:
        if not dataset_path:
            print("--dataset_path 또는 --dataset_url 중 하나는 반드시 지정해야 합니다.")
            sys.exit(1)
        if not os.path.exists(dataset_path):
            print(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")
            sys.exit(1)

    # 전처리 (원본 스크립트가 없으면 core.py에서 자동 건너뜀)
    _run([
        sys.executable,
        core_py,
        "preprocess",
        "--model_name",
        model_name,
        "--dataset_path",
        dataset_path,
        "--sample_rate",
        sample_rate,
        "--cpu_cores",
        cpu_cores,
    ])

    # 특징 추출 (원본 스크립트가 없으면 core.py에서 자동 건너뜀)
    _run([
        sys.executable,
        core_py,
        "extract",
        "--model_name",
        model_name,
        "--f0_method",
        f0_method,
        "--cpu_cores",
        cpu_cores,
        "--gpu",
        gpu,
        "--sample_rate",
        sample_rate,
        "--embedder_model",
        embedder_model,
        "--include_mutes",
        2,
    ])

    # 학습 (+ core.py 내부에서 인덱스 생성 호출)
    _run([
        sys.executable,
        core_py,
        "train",
        "--model_name",
        model_name,
        "--batch_size",
        batch_size,
        "--gpu",
        gpu,
        "--index_algorithm",
        index_algorithm,
    ])

    logs_dir = os.path.join("logs", model_name)
    print("==== DONE ====")
    print(f"- Logs:   {logs_dir}")
    print(f"- Index:  {os.path.join(logs_dir, model_name + '.index')}")


def main():
    p = argparse.ArgumentParser(description="RVC V2 one-shot runner (preprocess -> extract -> train)")
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--dataset_path", default="", type=str)
    p.add_argument("--dataset_url", default="", type=str)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--gpu", default="0", type=str)
    p.add_argument("--cpu_cores", default=os.cpu_count() or 4, type=int)
    p.add_argument("--index_algorithm", default="Auto", type=str)
    p.add_argument("--embedder_model", default="contentvec", type=str)
    p.add_argument("--f0_method", default="rmvpe", type=str)
    p.add_argument("--sample_rate", default=48000, type=int)
    args = p.parse_args()

    run_pipeline(
        model_name=args.model_name,
        dataset_path=args.dataset_path or None,
        dataset_url=args.dataset_url or None,
        batch_size=args.batch_size,
        gpu=args.gpu,
        cpu_cores=args.cpu_cores,
        index_algorithm=args.index_algorithm,
        embedder_model=args.embedder_model,
        f0_method=args.f0_method,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()


