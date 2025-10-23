import os
import torch
from collections import OrderedDict


def _extract_weight_dict(ckpt: dict) -> dict:
    """
    모델 체크포인트에서 가중치 dict만 추출
    - 한글 주석: RVC 포맷에서 'model' 키가 있을 수 있으므로 통합 처리
    """
    if "model" in ckpt:
        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in a.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = a[key]
        return opt
    else:
        # 이미 weight 포맷
        return {"weight": ckpt["weight"]}


def model_blender(name: str, path1: str, path2: str, ratio: float):
    """
    두 RVC 모델을 주어진 비율로 블렌딩하여 logs/{name}.pth 로 저장
    - 한글 주석: 샘플레이트/버전/설정값을 보존하고, 스피커 임베딩 차이가 다르면 최소 길이에 맞춰 혼합
    """
    try:
        message = f"Model {path1} and {path2} are merged with alpha {ratio}."
        ckpt1 = torch.load(path1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=True)

        sr1 = str(ckpt1["sr"]).lower().replace("k", "000")
        sr2 = str(ckpt2["sr"]).lower().replace("k", "000")
        if sr1 != sr2:
            print(
                f"Sample rate of {path1} {sr1} does not match the sample rate of {path2} {sr2}."
            )
            return "The sample rates of the two models are not the same."

        # 메타 데이터 보존
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = sr1
        vocoder = ckpt1.get("vocoder", "HiFi-GAN")

        w1 = _extract_weight_dict(ckpt1)["weight"]
        w2 = _extract_weight_dict(ckpt2)["weight"]

        if sorted(list(w1.keys())) != sorted(list(w2.keys())):
            return "Fail to merge the models. The model architectures are not the same."

        opt = OrderedDict()
        opt["weight"] = {}
        for key in w1.keys():
            if key == "emb_g.weight" and w1[key].shape != w2[key].shape:
                min_shape0 = min(w1[key].shape[0], w2[key].shape[0])
                opt["weight"][key] = (
                    ratio * (w1[key][:min_shape0].float())
                    + (1 - ratio) * (w2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    ratio * (w1[key].float()) + (1 - ratio) * (w2[key].float())
                ).half()

        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["info"] = message
        opt["vocoder"] = vocoder

        os.makedirs("logs", exist_ok=True)
        save_path = os.path.join("logs", f"{name}.pth")
        torch.save(opt, save_path)
        print(message)
        return message, save_path
    except Exception as error:
        print(f"An error occurred blending the models: {error}")
        return error


