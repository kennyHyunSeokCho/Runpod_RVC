import os
import sys
import argparse
import subprocess

python = sys.executable


def run_preprocess(model_name: str, dataset_path: str, sample_rate: int, cpu_cores: int):
    """
    데이터 전처리. 원본 스크립트(rvc/train/preprocess/preprocess.py)에 의존
    - 한글 주석: v2_core만 업로드 시 이 단계는 건너뛰어야 함
    """
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
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
    extract_script = os.path.join("rvc", "train", "extract", "extract.py")
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
    else:
        print("사용법: preprocess | extract | train | index 서브커맨드 중 하나를 선택하세요.")


if __name__ == "__main__":
    main()


