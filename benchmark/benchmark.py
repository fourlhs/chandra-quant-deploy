import argparse
import asyncio
import base64
import csv
import io
import json
import time
from pathlib import Path

import pynvml
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from pdf2image import convert_from_path
from tabulate import tabulate

OCR_PROMPT = "Extract all text from this document."
MAX_TOKENS = 4096


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())


def cer(reference: str, hypothesis: str) -> float:
    import editdistance
    if not reference:
        return 0.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def pdf_page_to_b64(pdf_path: str, page_number: int, dpi: int = 150) -> str:
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
    if not pages:
        raise ValueError(f"Could not load page {page_number} from {pdf_path}")
    buf = io.BytesIO()
    pages[0].convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def load_samples(pages: list, gt_dir: Path, base_dir: Path) -> list:
    samples = []
    print("Pre-loading images and ground truth...")
    for entry in pages:
        pdf_path = base_dir / entry["pdf_path"]
        gt_file  = gt_dir  / entry["markdown_file"]

        if not pdf_path.exists():
            print(f"  [skip] PDF not found: {pdf_path}")
            continue
        if not gt_file.exists():
            print(f"  [skip] GT not found: {gt_file}")
            continue
        try:
            b64 = pdf_page_to_b64(str(pdf_path), entry["page_number"])
        except Exception as e:
            print(f"  [skip] {entry['id']}: {e}")
            continue

        gt_text = html_to_text(gt_file.read_text(encoding="utf-8"))
        samples.append({"id": entry["id"], "b64": b64, "gt_text": gt_text})
        print(f"  loaded {entry['id']}")

    print(f"Loaded {len(samples)} samples.\n")
    return samples


def read_vram_gib(gpu_index: int) -> float:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**3


async def send_one(client: AsyncOpenAI, sample: dict, model_name: str):
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sample['b64']}"}},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    text   = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return text, tokens


async def run_batches(api_url: str, model_name: str, samples: list, batch_size: int) -> list:
    client  = AsyncOpenAI(base_url=f"{api_url}/v1", api_key="none")
    results = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        ids   = [s["id"] for s in batch]

        print(f"  batch {i//batch_size + 1}: {ids} ...", end=" ", flush=True)
        t0      = time.perf_counter()
        outputs = await asyncio.gather(*[send_one(client, s, model_name) for s in batch])
        elapsed = time.perf_counter() - t0

        per_page_latency = elapsed / len(batch)
        batch_tokens = 0
        for j, (text, tokens) in enumerate(outputs):
            pred = " ".join(text.split())
            batch_tokens += tokens
            error_rate = cer(batch[j]["gt_text"], pred)
            results.append({
                "id":        batch[j]["id"],
                "cer":       error_rate,
                "latency_s": per_page_latency,
                "tokens":    tokens,
            })
            print(f"{batch[j]['id']} CER={error_rate:.3f} tok={tokens}", end="  ")
        tps = batch_tokens / elapsed if elapsed > 0 else 0
        print(f"({elapsed:.1f}s  {tps:.0f} tok/s)")

    return results


def run_benchmark(api_url: str, model_name: str, samples: list, batch_size: int, gpu_index: int):
    print(f"\n{'='*60}")
    print(f"  batch_size={batch_size}")
    print(f"{'='*60}")

    vram_gib = read_vram_gib(gpu_index)
    print(f"  VRAM in use: {vram_gib:.2f} GiB\n")

    results = asyncio.run(run_batches(api_url, model_name, samples, batch_size))
    return results, vram_gib


def summarise(model_label: str, batch_size: int, results: list, vram_gib: float) -> dict:
    if not results:
        return {"model": model_label, "batch_size": batch_size, "pages": 0,
                "avg_cer": None, "avg_latency_s": None, "throughput_pps": None,
                "total_tokens": None, "tokens_per_sec": None, "total_time_s": None,
                "vram_gib": vram_gib}
    avg_cer      = sum(r["cer"] for r in results) / len(results)
    avg_latency  = sum(r["latency_s"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    total_time   = avg_latency * len(results)
    tokens_per_s = total_tokens / total_time if total_time > 0 else 0
    return {
        "model":          model_label,
        "batch_size":     batch_size,
        "pages":          len(results),
        "avg_cer":        round(avg_cer, 4),
        "avg_latency_s":  round(avg_latency, 2),
        "throughput_pps": round(1 / avg_latency, 3),
        "total_tokens":   total_tokens,
        "tokens_per_sec": round(tokens_per_s, 1),
        "total_time_s":   round(total_time, 1),
        "vram_gib":       round(vram_gib, 2),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    required=True)
    p.add_argument("--gt-dir",      required=True)
    p.add_argument("--base-dir",    required=True)
    p.add_argument("--model",       default="model",   help="label for this run, e.g. fp8 or int4")
    p.add_argument("--api-url",     default="http://localhost:8000")
    p.add_argument("--model-name",  default="chandra", help="served_model_name in vLLM")
    p.add_argument("--batch-sizes", default="4,8,20")
    p.add_argument("--gpu-index",   type=int, default=0)
    p.add_argument("--output",      default="results.csv")
    return p.parse_args()


def main():
    args        = parse_args()
    gt_dir      = Path(args.gt_dir).expanduser()
    base_dir    = Path(args.base_dir).expanduser()
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    api_url     = args.api_url.rstrip("/")

    with open(args.manifest) as f:
        manifest = json.load(f)

    samples = load_samples(manifest["pages"], gt_dir, base_dir)

    summaries = []
    all_rows  = []

    for batch_size in batch_sizes:
        results, vram_gib = run_benchmark(api_url, args.model_name, samples, batch_size, args.gpu_index)
        summary = summarise(args.model, batch_size, results, vram_gib)
        summaries.append(summary)

        for r in results:
            all_rows.append({
                "model":      args.model,
                "batch_size": batch_size,
                "id":         r["id"],
                "cer":        r["cer"],
                "latency_s":  r["latency_s"],
                "tokens":     r["tokens"],
                "vram_gib":   vram_gib,
            })

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    headers = ["Model", "Batch", "Pages", "Avg CER ↓", "Latency/page (s) ↓",
               "Throughput (p/s) ↑", "Total Tokens", "Tokens/sec ↑", "Total Time (s)", "VRAM (GiB)"]
    rows = [[
        s["model"],
        s["batch_size"],
        s["pages"],
        f"{s['avg_cer']:.4f}"        if s["avg_cer"]        is not None else "—",
        f"{s['avg_latency_s']:.2f}"  if s["avg_latency_s"]  is not None else "—",
        f"{s['throughput_pps']:.3f}" if s["throughput_pps"] is not None else "—",
        s["total_tokens"]            if s["total_tokens"]    is not None else "—",
        f"{s['tokens_per_sec']:.1f}" if s["tokens_per_sec"] is not None else "—",
        f"{s['total_time_s']:.1f}"   if s["total_time_s"]   is not None else "—",
        f"{s['vram_gib']:.2f}"       if s["vram_gib"]       is not None else "—",
    ] for s in summaries]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    if all_rows:
        out = Path(args.output)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDetailed results saved to: {out}")


if __name__ == "__main__":
    main()