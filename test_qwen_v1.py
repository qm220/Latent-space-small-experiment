from pathlib import Path
import json

import torch
import transformers
from PIL import Image
from huggingface_hub import model_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_PATH = Path("data/images/part.png")
OUT = Path("runs/smoke_test")
OUT.mkdir(parents=True, exist_ok=True)

assert IMAGE_PATH.exists(), f"Missing image: {IMAGE_PATH}"
assert torch.cuda.is_available(), "CUDA is unavailable."

torch.manual_seed(42)

# Record the exact checkpoint used.
revision = model_info(MODEL_ID).sha

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=revision,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    revision=revision,
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
)
model.eval()

# Limit image size for the initial installation test.
image = Image.open(IMAGE_PATH).convert("RGB")
image.thumbnail((672, 672))
image.save(OUT / "input_image.png")

prompt = (
    "Describe the visible geometry of this mechanical part. "
    "Then suggest its possible function. "
    "Clearly distinguish visual observations from assumptions."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
).to("cuda:0")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
    )

new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
)[0]

print("\nMODEL RESPONSE:\n")
print(answer)

(OUT / "answer.txt").write_text(answer, encoding="utf-8")
(OUT / "metadata.json").write_text(
    json.dumps(
        {
            "model": MODEL_ID,
            "revision": revision,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "image_size": list(image.size),
            "prompt": prompt,
            "do_sample": False,
            "max_new_tokens": 160,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nResults saved in:", OUT.resolve())