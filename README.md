# chandra-quant-deploy

Quantization, benchmarking, and vLLM deployment pipeline for [Chandra OCR](https://huggingface.co/datalab-to/chandra) (Qwen3-VL-8B).

Supports FP8 and INT4 quantization with no calibration data needed. Includes a benchmark suite to compare accuracy (CER), speed, and VRAM across model variants.

## Requirements

- NVIDIA GPU with FP8 support (Blackwell or Hopper — RTX 5090, H100, H200, B200)
- CUDA 12.8+
- Docker + NVIDIA Container Toolkit (for serving)

## Setup

```bash
cp local.env.example local.env
# edit local.env with your paths
```

## 1. Download the model

```bash
pip install huggingface_hub
python download-hf-model.py
```

## 2. Quantize

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r quantize/requirements.txt

# FP8 (~8GB, recommended)
python quantize/quantize.py

# INT4 (~4GB, more aggressive)
python quantize/quantize_int4.py
```

Both scripts require no calibration data. Vision encoder stays in BF16 in both cases.

## 3. Benchmark

Compare BF16 vs FP8 vs INT4 on accuracy (CER), latency, throughput, and VRAM:

```bash
pip install -r benchmark/requirements.txt
python benchmark/benchmark.py \
  --manifest /path/to/manifest.json \
  --gt-dir   /path/to/ground_truth_html \
  --base-dir /path/to/data_root \
  --output   results.csv
```

## 4. Serve

```bash
cd serve
# create serve/.env — set OUTPUT_PATH to the model you want to serve
docker compose up -d
```

The server runs on port 8000 with an OpenAI-compatible API.

## 5. Run OCR

```bash
pip install -r serve/requirements.txt
python serve/client.py document.pdf
python serve/client.py scan.png --output result.txt
python serve/client.py *.pdf --host YOUR_SERVER_IP
```

## Structure

```
quantize/   quantization scripts (FP8, INT4)
benchmark/  benchmark suite (CER, latency, VRAM)
serve/      Docker + vLLM serving stack + OCR client
```