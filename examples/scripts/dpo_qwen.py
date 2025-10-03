# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "Pillow>=9.4.0",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Without dataset streaming:

```
accelerate launch examples/scripts/dpo_qwen.py \
    --model_name_or_path /home/ma-user/work/pretrain_models/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir saving_dir \
    --gradient_checkpointing 1 \
    --max_prompt_length 2048 \
    --max_length 4096

```

With dataset streaming:

```
accelerate launch examples/scripts/dpo_vlm.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 2 \
    --max_steps 100 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_idefics_rlaif-v \
    --dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules all-linear
```

accelerate launch examples/scripts/dpo_qwen.py     --model_name_or_path /home/ma-user/work/pretrain_models/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct     --per_device_train_batch_size 1     --gradient_accumulation_steps 32     --dataset_num_proc 64     --output_dir saving_dir     --gradient_checkpointing 1     --max_prompt_length 4096     --max_length 8192 --deepspeed examples/accelerate_configs/ds_config_zero3.json
"""

import os

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
from datasets import load_dataset, Image, Sequence


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 与梯度检查点搭配：禁用缓存
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, do_image_splitting=True
    )
    tokenizer = processor.tokenizer

    # Set up the chat template
    if model.config.model_type == "idefics2":
        pass  # the processor already has a valid chat template
    elif model.config.model_type == "paligemma":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    elif model.config.model_type == "llava":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    data_files = {
        "train": "/home/ma-user/work/wbd/01_projects/clas/HKEmodel/Qwen25VL/mmsd_v2_dpo.parquet",
    }
    dsdict = load_dataset("parquet", data_files=data_files)
    dataset = dsdict["train"]

    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    training_args.remove_unused_columns=False
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)