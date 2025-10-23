import glob
import json
import os
import numpy as np
import soundfile as sf


def read_precision_config(config_dir: str) -> str:
    """
    assets/config.json 에서 precision 값을 읽어옴
    - 한글 주석: 파일이 없거나 키가 없으면 None 반환
    """
    try:
        with open(os.path.join(config_dir, "assets", "config.json"), "r") as f:
            data = json.load(f)
            return data.get("precision")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def list_wavs(sliced_dir: str):
    """
    슬라이스된 wav 리스트 반환
    """
    return glob.glob(os.path.join(sliced_dir, "*.wav"))


def read_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
    except FileNotFoundError:
        return None


def write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def ensure_mute_assets(logs_dir: str, sample_rate: int = 48000, seconds: float = 1.0, min_files: int = 2):
    """
    logs 디렉토리에 무음 wav 파일이 필요한 최소 개수만큼 준비
    - 한글 주석: extract 단계의 include_mutes 대비 안전장치
    """
    targets = [
        os.path.join(logs_dir, "mute"),
        os.path.join(logs_dir, "mute_spin"),
        os.path.join(logs_dir, "mute_spin-v2"),
    ]
    # 패키지된 무음 자산이 있으면 우선 복사
    packaged = [
        os.path.join("logs", "mute"),
        os.path.join("logs", "mute_spin"),
        os.path.join("logs", "mute_spin-v2"),
    ]
    num_samples = int(sample_rate * seconds)
    silence = np.zeros((num_samples,), dtype=np.float32)

    for src, tgt in zip(packaged, targets):
        if os.path.isdir(src):
            os.makedirs(tgt, exist_ok=True)
            for wav in glob.glob(os.path.join(src, "*.wav")):
                try:
                    base = os.path.basename(wav)
                    dst = os.path.join(tgt, base)
                    if not os.path.exists(dst):
                        # 단순 복사
                        data, sr = sf.read(wav, dtype="float32")
                        if sr != sample_rate:
                            # 리샘플 필요하면 생략하고 합성으로 대체
                            pass
                        else:
                            sf.write(dst, data, sr)
                except Exception:
                    pass

    for tgt in targets:
        os.makedirs(tgt, exist_ok=True)
        # 현재 wav 개수 확인
        existing = glob.glob(os.path.join(tgt, "*.wav"))
        need = max(0, min_files - len(existing))
        for i in range(need):
            out_path = os.path.join(tgt, f"mute_{i+1}.wav")
            try:
                sf.write(out_path, silence, sample_rate)
            except Exception:
                # 실패해도 전체 파이프라인 진행에는 영향 없도록 함
                pass


