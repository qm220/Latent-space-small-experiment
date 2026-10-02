from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from huggingface_hub import model_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from knowledge_inv.model.activations import SequenceActivationCapture, pack_sequence_activations
from knowledge_inv.paths import load_project_config

EIGHT_B_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
EIGHT_B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SUPPORTED_QUANTIZATION = {None, "nf4", "fp4", "prequantized"}

VIEW_CAPTIONS = {
    "front": "Front view",
    "back": "Back view",
    "left": "Left view",
    "right": "Right view",
    "top": "Top view",
    "bottom": "Bottom view",
    "toprightiso": "Isometric view (top-right)",
}


def as_image_list(images: Image.Image | Sequence[Image.Image]) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images]
    return list(images)


def resolve_revision(model_id: str, fallback: str | None = None) -> str:
    try:
        return model_info(model_id).sha
    except Exception:
        if fallback:
            return fallback
        raise


def load_image(path: str | Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def build_bnb_4bit_config(
    quant_type: str,
    skip_modules: Sequence[str] | None = None,
) -> BitsAndBytesConfig:
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError(f"quant_type must be nf4 or fp4, got {quant_type!r}")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "4-bit loading needs bitsandbytes. Install it into .venv, e.g. "
            "`pip install bitsandbytes`."
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(skip_modules) if skip_modules is not None else ["visual"],
    )


class QwenVLRunner:
    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
        quantization: str | None = None,
        skip_modules: Sequence[str] | None = None,
        revision: str | None = None,
        device_map: str | dict | None = None,
        local_files_only: bool | None = None,
        cache_dir: str | Path | None = None,
    ):
        config = load_project_config()
        self.model_id = model_id or config["model_id"]
        self.device = device
        self.attn_implementation = attn_implementation
        if quantization not in SUPPORTED_QUANTIZATION:
            raise ValueError(
                f"quantization must be one of {sorted(x for x in SUPPORTED_QUANTIZATION if x)}, "
                f"or None for BF16. Got {quantization!r}."
            )
        self.quantization = quantization
        if not quantization:
            self.skip_modules = []
        elif skip_modules is None:
            self.skip_modules = ["visual"]
        else:
            self.skip_modules = list(skip_modules)
        if local_files_only is None:
            self.local_files_only = os.environ.get("HF_HUB_OFFLINE", "") in {"1", "true", "True"}
        else:
            self.local_files_only = bool(local_files_only)
        self.device_map = device_map if device_map is not None else {"": device}
        fallback = EIGHT_B_REVISION if self.model_id == EIGHT_B_MODEL_ID else None
        if revision:
            self.revision = revision
        elif self.local_files_only:
            self.revision = None
        else:
            self.revision = resolve_revision(self.model_id, fallback=fallback)
        processor_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
        }
        if self.revision:
            processor_kwargs["revision"] = self.revision
        if cache_dir:
            processor_kwargs["cache_dir"] = str(cache_dir)
        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        resolved_device_map: str | dict = (
            device_map if device_map is not None else {"": device}
        )
        load_kwargs: dict[str, Any] = {
            "device_map": resolved_device_map,
            "attn_implementation": attn_implementation,
            "local_files_only": self.local_files_only,
        }
        if self.revision:
            load_kwargs["revision"] = self.revision
        if cache_dir:
            load_kwargs["cache_dir"] = str(cache_dir)
        if quantization in {"nf4", "fp4"}:
            load_kwargs["quantization_config"] = build_bnb_4bit_config(
                quantization,
                skip_modules=self.skip_modules,
            )
        elif quantization == "prequantized":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "This checkpoint still needs bitsandbytes at runtime. "
                    "Install it into .venv, e.g. `pip install bitsandbytes`."
                ) from exc
        else:
            load_kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        self.model.eval()
        language = self.model.model.language_model
        self.n_layers = len(language.layers)
        self.hidden_size = int(getattr(language.config, "hidden_size", 0) or 0)
        self.layer_names = [f"model.model.language_model.layers.{idx}" for idx in range(self.n_layers)]

    def _tensors_device(self) -> torch.device:
        try:
            weight = self.model.get_input_embeddings().weight
            return weight.device
        except Exception:
            return torch.device(self.device)

    def quantization_inventory(self) -> dict[str, Any]:
        """Count 4-bit vs leftover high-precision tensors after load."""
        fourbit_linears = 0
        fp_linears: list[str] = []
        bytes_by_dtype: dict[str, int] = {}
        devices: dict[str, int] = {}
        for name, module in self.model.named_modules():
            cls = type(module).__name__
            if cls == "Linear4bit":
                fourbit_linears += 1
            elif cls == "Linear":
                fp_linears.append(name)
        for name, param in self.model.named_parameters():
            dtype = str(param.dtype).replace("torch.", "")
            bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + param.numel() * param.element_size()
            dev = str(param.device)
            devices[dev] = devices.get(dev, 0) + 1
        vision_still_fp = [n for n in fp_linears if ".visual." in n or n.startswith("model.visual")]
        language_still_fp = [n for n in fp_linears if n not in vision_still_fp]
        return {
            "scheme": self.quantization,
            "skip_modules": self.skip_modules,
            "n_linear4bit": fourbit_linears,
            "n_unquantized_linear": len(fp_linears),
            "unquantized_linear_names": fp_linears[:40],
            "unquantized_linear_truncated": max(0, len(fp_linears) - 40),
            "vision_unquantized_linear": len(vision_still_fp),
            "language_unquantized_linear": len(language_still_fp),
            "parameter_bytes_gb": {k: round(v / 1024**3, 3) for k, v in sorted(bytes_by_dtype.items())},
            "parameter_devices": devices,
            "all_parameters_on_cuda": all(str(dev).startswith("cuda") for dev in devices),
            "note": (
                "Linear4bit modules hold 4-bit weights in VRAM. Matmuls still dequantize a tile "
                "to bfloat16 (compute dtype); that is not a second unquantized copy of the model. "
                "Embeddings, RMSNorm, and KV cache stay high precision."
            ),
        }

    def prepare_inputs(
        self,
        images: Image.Image | Sequence[Image.Image],
        prompt: str,
        view_names: Sequence[str] | None = None,
    ) -> tuple[dict, str]:
        image_list = as_image_list(images)
        if not image_list:
            raise ValueError("At least one image is required.")
        names = list(view_names) if view_names is not None else [None] * len(image_list)
        if len(names) != len(image_list):
            raise ValueError(f"view_names length {len(names)} != n_images {len(image_list)}")
        content: list[dict[str, Any]] = []
        for name in names:
            caption = VIEW_CAPTIONS.get(name or "", name) if name else None
            if caption:
                content.append({"type": "text", "text": f"{caption}:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=image_list,
            return_tensors="pt",
        ).to(self._tensors_device())
        return inputs, text

    def generate(
        self,
        images: Image.Image | Sequence[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        capture_layers: list[int] | None = None,
        view_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        image_list = as_image_list(images)
        inputs, rendered_prompt = self.prepare_inputs(image_list, prompt, view_names=view_names)
        prompt_length = int(inputs["input_ids"].shape[1])
        activations = None
        capture = SequenceActivationCapture(self.model, capture_layers) if capture_layers else None
        try:
            if capture is not None:
                capture.register()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            if capture is not None:
                full_ids = output_ids[0].detach().to("cpu").tolist()
                tokenizer = getattr(self.processor, "tokenizer", None)
                token_strings = (
                    tokenizer.convert_ids_to_tokens(full_ids) if tokenizer is not None else [""] * len(full_ids)
                )
                activations = pack_sequence_activations(
                    stacked=capture.stacked(),
                    layer_names=capture.layer_names,
                    layer_indices=capture_layers,
                    token_ids=full_ids,
                    token_strings=token_strings,
                    prompt_length=prompt_length,
                )
        finally:
            if capture is not None:
                capture.close()
        new_tokens = output_ids[:, prompt_length:]
        answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        elapsed = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()
        return {
            "answer": answer.strip(),
            "rendered_prompt": rendered_prompt,
            "prompt": prompt,
            "runtime_sec": round(elapsed, 3),
            "peak_gpu_memory_bytes": int(peak_bytes),
            "peak_gpu_memory_gb": round(peak_bytes / 1024**3, 3),
            "input_token_count": prompt_length,
            "output_token_count": int(new_tokens.shape[1]),
            "n_images": len(image_list),
            "view_names": list(view_names) if view_names is not None else None,
            "output_token_ids": new_tokens[0].detach().to("cpu").tolist(),
            "activations": activations,
            "settings": {
                "model_id": self.model_id,
                "revision": self.revision,
                "dtype": "bfloat16",
                "n_language_layers": self.n_layers,
                "hidden_size": self.hidden_size,
                "device": self.device,
                "attn_implementation": self.attn_implementation,
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "quantization": self.quantization,
                "quantization_skip_modules": self.skip_modules,
                "quantization_note": (
                    None
                    if self.quantization is None
                    else (
                        "Language Linear weights are 4-bit; compute and residual-stream "
                        "activations stay bfloat16. This is bitsandbytes NF4/FP4, not NVIDIA NVFP4."
                        if self.quantization in {"nf4", "fp4"}
                        else "Loaded a checkpoint that was already 4-bit quantized."
                    )
                ),
            },
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
