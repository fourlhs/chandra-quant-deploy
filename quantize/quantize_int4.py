import os
import torch
from dotenv import load_dotenv
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "local.env"))

MODEL_PATH  = os.getenv("MODEL_CHECKPOINT", "./chandra")
OUTPUT_PATH = os.getenv("OUTPUT_PATH_INT4",  "./chandra-int4")

print(f"Loading model from: {MODEL_PATH}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Vision encoder and lm_head stay in BF16; everything else is INT4.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*visual.*"],
)

print("Running INT4 W4A16 quantization...")
oneshot(model=model, recipe=recipe)

print(f"Saving quantized model to: {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH)
processor.save_pretrained(OUTPUT_PATH)
print("Done.")
print()
print("Deploy with vLLM:")
print(f"  vllm serve {OUTPUT_PATH}")