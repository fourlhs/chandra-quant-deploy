"""
Prepares calibration dataset for Chandra FP8 quantization.
Accepts a folder of PDFs and/or images, formats them as
Qwen3-VL chat inputs for use with llm-compressor.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


OCR_PROMPTS = [
    "Extract all text from this document.",
    "Perform OCR on this image and return all text.",
    "Read and transcribe all text visible in this image.",
    "What does this document say? Return all text.",
]


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list:
    from pdf2image import convert_from_path
    images = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"  {pdf_path.name}: {len(images)} pages")
    return images


def collect_images(data_dir: str) -> list:
    data_dir = Path(data_dir)
    images = []

    for pdf_path in sorted(data_dir.glob("**/*.pdf")):
        images.extend(pdf_to_images(pdf_path))

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.webp"):
        for img_path in sorted(data_dir.glob(f"**/{ext}")):
            images.append(Image.open(img_path).convert("RGB"))

    return images


class CalibrationDataset(Dataset):
    def __init__(self, images: list, processor, max_seq_len: int = 2048):
        self.samples = self._prepare(images, processor, max_seq_len)

    def _prepare(self, images: list, processor, max_seq_len: int) -> list:
        samples = []
        for i, image in enumerate(images):
            prompt = OCR_PROMPTS[i % len(OCR_PROMPTS)]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            samples.append({k: v.squeeze(0) for k, v in inputs.items()})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def load_calibration_data(
    data_dir: str,
    processor,
    num_samples: int = 512,
    max_seq_len: int = 2048,
) -> CalibrationDataset:
    print(f"Loading calibration images from: {data_dir}")
    images = collect_images(data_dir)

    if not images:
        raise ValueError(
            f"No PDFs or images found in '{data_dir}'.\n"
            "Add documents representative of your real OCR workload."
        )

    print(f"Found {len(images)} source images.")

    if len(images) < num_samples:
        images = (images * (num_samples // len(images) + 1))[:num_samples]
    else:
        images = images[:num_samples]

    print(f"Preparing {len(images)} calibration samples...")
    return CalibrationDataset(images, processor, max_seq_len)
