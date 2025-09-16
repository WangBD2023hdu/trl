from __future__ import annotations
import os, io
from pathlib import Path

import torch
from datasets import load_dataset, Image, Sequence
from PIL import Image as PILImage

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    # 如果用 Trainer 的 TrainingArguments，也可从 transformers 导入
)
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# ---------- 建议：固定 NCCL 环境 ----------
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # 如仍不稳，可再打开（降性能但更稳）

# ---------- 必做：绑定本进程 GPU ----------
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

def _ensure_pil(img_item):
    """把 datasets 的 Image(decode=True/False) 项/字节流转成 PIL"""
    if isinstance(img_item, dict):
        b = img_item.get("bytes")
        p = img_item.get("path")
        if b is not None:
            return PILImage.open(io.BytesIO(b)).convert("RGB")
        if p:
            return PILImage.open(p).convert("RGB")
    if isinstance(img_item, PILImage.Image):
        return img_item
    raise ValueError(f"bad image item: {type(img_item)} | {img_item}")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # --------------- Model & Tokenizer ---------------
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 主模型（与参考模型同架构）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    # 与梯度检查点搭配：禁用缓存
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 只开一处梯度检查点，且使用 non-reentrant
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 参考模型与主模型同架构，并冻结参数
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        **({} if quantization_config is None else {}),
        **model_kwargs,
    )
    ref_model.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        do_image_splitting=False,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # --------------- Dataset ---------------
    data_files = {
        "train": "/home/ma-user/work/wbd/01_projects/clas/HKEmodel/Qwen25VL/test.parquet",
    }
    dsdict = load_dataset("parquet", data_files=data_files)
    dataset = dsdict["train"]

    # 若是“列表图像”列
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(Image(decode=True)))
    elif "image" in dataset.column_names:
        dataset = dataset.cast_column("image", Image(decode=True))

    # 预清洗坏样本
    def safe_check(ex):
        try:
            if "images" in ex and ex["images"]:
                _ = _ensure_pil(ex["images"][0])
            if "image" in ex:
                _ = _ensure_pil(ex["image"])
            # 按你的 schema 调整：DPO 需要 prompt/chosen/rejected
            assert "prompt" in ex and "chosen" in ex and "rejected" in ex
            return True
        except Exception:
            return False

    dataset = dataset.filter(safe_check, desc="filter_bad_examples")

    # --------------- Training Args 关键开关 ---------------
    # 更稳的 DDP 设置：关闭“查找未用参数”，并开启非重入 checkpoint
    setattr(training_args, "ddp_find_unused_parameters", False)
    setattr(training_args, "gradient_checkpointing", True)
    # 新版 transformers 支持设置 kwargs
    setattr(
        training_args,
        "gradient_checkpointing_kwargs",
        {"use_reentrant": False},
    )
    # 如果你的前向图稳定（每步结构一致），可考虑静态图（见下 Trainer 初始化后注释）

    # 根据环境保守一些
    if not hasattr(training_args, "dataloader_num_workers"):
        training_args.dataloader_num_workers = 0
    else:
        training_args.dataloader_num_workers = 0
    setattr(training_args, "torch_compile", False)

    # --------------- Trainer ---------------
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,    # 必须是 Dataset
        eval_dataset=None,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    # （可选）静态图：仅当前向图确定不变时使用
    try:
        ddp_model = trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(ddp_model, "_set_static_graph"):
            pass
            # ddp_model._set_static_graph()  # 前向图稳定时可开启
    except Exception:
        pass

    trainer.train()

    # Save and (optionally) push
    trainer.save_model(training_args.output_dir)
    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name)