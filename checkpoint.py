import os
import torch
from rvc.v2_core.vendor.utils import latest_checkpoint_path, load_checkpoint


def try_resume_or_load(pretrainG, pretrainD, experiment_dir, net_g, net_d, optim_g, optim_d, rank):
    """
    최신 체크포인트에서 재개하거나 사전학습 가중치 로드
    - 한글 주석: 재개 성공시 다음 epoch 시작 번호 반환, 아니면 1
    """
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        return epoch_str
    except Exception:
        if pretrainG not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            ckpt = torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(ckpt)
            else:
                net_g.load_state_dict(ckpt)
            del ckpt

        if pretrainD not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            ckpt = torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(ckpt)
            else:
                net_d.load_state_dict(ckpt)
            del ckpt
        return 1


