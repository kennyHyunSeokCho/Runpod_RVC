import os
from rvc.v2_core.vendor.extract_model import extract_model


def export_models_if_needed(model_add, net_g, experiment_dir, model_name, epoch, global_step, hps, overtrain_info, sr, vocoder):
    """
    스케줄에 따라 합성된 모델(.pth) 내보내기
    - 한글 주석: 파일이 존재하지 않으면 추출 수행
    """
    if not model_add:
        return
    ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
    for m in model_add:
        if not os.path.exists(m):
            extract_model(
                ckpt=ckpt,
                sr=sr,
                name=model_name,
                model_path=m,
                epoch=epoch,
                step=global_step,
                hps=hps,
                overtrain_info=overtrain_info,
                vocoder=vocoder,
            )


