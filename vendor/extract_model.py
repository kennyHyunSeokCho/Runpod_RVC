import datetime
import hashlib
import json
import os
from collections import OrderedDict

import torch

from rvc.v2_core.vendor.utils import replace_keys_in_dict


def extract_model(
    ckpt,
    sr,
    name,
    model_path,
    epoch,
    step,
    hps,
    overtrain_info,
    vocoder,
    pitch_guidance=True,
    version="v2",
):
    try:
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        if os.path.exists(os.path.join(model_dir, "model_info.json")):
            with open(os.path.join(model_dir, "model_info.json"), "r") as f:
                data = json.load(f)
                dataset_length = data.get("total_dataset_duration", None)
                embedder_model = data.get("embedder_model", None)
                speakers_id = data.get("speakers_id", 1)
        else:
            dataset_length = None
            embedder_model = None
            speakers_id = 1

        # author 정보는 없으면 None 처리
        try:
            with open(os.path.join(os.getcwd(), "assets", "config.json"), "r") as f:
                data = json.load(f)
                model_author = data.get("model_author", None)
        except Exception:
            model_author = None

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = pitch_guidance
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = f"{name}-{epoch}-{step}-{sr}-{version}-{opt['config']}"
        opt["model_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["overtrain_info"] = overtrain_info
        opt["dataset_length"] = dataset_length
        opt["model_name"] = name
        opt["author"] = model_author
        opt["embedder_model"] = embedder_model
        opt["speakers_id"] = speakers_id
        opt["vocoder"] = vocoder

        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(
                    opt, ".parametrizations.weight.original1", ".weight_v"
                ),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            model_path,
        )

        print(f"Saved model '{model_path}' (epoch {epoch} and step {step})")

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")


