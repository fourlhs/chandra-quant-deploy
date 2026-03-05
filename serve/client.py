import argparse
import base64
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv(Path(__file__).parent.parent / "local.env")

DEFAULT_HOST   = os.getenv("SERVE_HOST",  "localhost")
DEFAULT_PORT   = int(os.getenv("SERVE_PORT", "8000"))
DEFAULT_PROMPT = os.getenv("OCR_PROMPT",  "Extract all text from this document.")
MODEL_NAME     = "chandra"


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def load_pages(path: Path) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pdf2image import convert_from_path
        return convert_from_path(str(path), dpi=150)
    else:
        return [Image.open(path).convert("RGB")]


def ocr_image(client: OpenAI, img: Image.Image, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": image_to_data_uri(img)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def parse_args():
    p = argparse.ArgumentParser(description="Chandra OCR client")
    p.add_argument("files", nargs="+", type=Path, help="Image or PDF files to OCR")
    p.add_argument("--host",   default=DEFAULT_HOST)
    p.add_argument("--port",   default=DEFAULT_PORT, type=int)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", "-o", help="Write results to this file (default: print to stdout)")
    return p.parse_args()


def main():
    args = parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="none",
    )

    results = []

    for file_path in args.files:
        if not file_path.exists():
            print(f"[skip] {file_path} not found", file=sys.stderr)
            continue

        pages = load_pages(file_path)
        print(f"[{file_path.name}] {len(pages)} page(s)", file=sys.stderr)

        for i, page in enumerate(pages, 1):
            print(f"  page {i}/{len(pages)}...", file=sys.stderr, end=" ", flush=True)
            text = ocr_image(client, page, args.prompt)
            print("done", file=sys.stderr)
            header = f"### {file_path.name}" + (f" — page {i}" if len(pages) > 1 else "")
            results.append(f"{header}\n\n{text}")

    combined = "\n\n---\n\n".join(results)

    if args.output:
        Path(args.output).write_text(combined, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(combined)


if __name__ == "__main__":
    main()