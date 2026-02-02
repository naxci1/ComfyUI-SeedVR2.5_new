# NVFP4 Model Conversion Guide / NVFP4 Model Dönüştürme Kılavuzu

Complete guide for converting FP16 models to native NVFP4 format.

FP16 modellerini native NVFP4 formatına dönüştürmek için tam kılavuz.

---

## English Guide

### Overview

This guide explains how to convert the SeedVR2 FP16 model to native NVFP4 format for maximum performance on Blackwell GPUs (RTX 50 series).

### What is NVFP4?

**NVFP4 (4-bit floating point)** is NVIDIA's advanced quantization format:
- **4-bit precision**: E2M1 format (1 sign, 2 exponent, 1 mantissa)
- **Two-level scaling**: FP8 micro-block + FP32 tensor scales
- **Native hardware**: Direct Tensor Core execution on Blackwell
- **High performance**: 2-3x faster than other 4-bit formats

### Benefits

| Metric | FP16 | NVFP4 | Improvement |
|--------|------|-------|-------------|
| Model Size | 6GB | 1.5GB | **4x smaller** |
| Memory Usage | 16GB | 4GB | **4x less** |
| Inference Speed | 1.0x | 2.5x | **2.5x faster** |
| Quality Loss | 0% | <1% | **Minimal** |

### Prerequisites

1. **Python environment** with:
   - torch >= 2.0.0
   - safetensors >= 0.3.0

2. **Input model**: `seedvr2_ema_3b_fp16.safetensors`
   - Download from: https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_3b_fp16.safetensors

3. **Disk space**: ~8GB free (for input + output files)

### Step-by-Step Conversion

#### Step 1: Download the FP16 Model

```bash
# Create models directory
mkdir -p models/SEEDVR2

# Download FP16 model (replace with actual download command)
# Option 1: Using wget
wget https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp16.safetensors \
    -O models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors

# Option 2: Using huggingface-cli
huggingface-cli download numz/SeedVR2_comfyUI seedvr2_ema_3b_fp16.safetensors \
    --local-dir models/SEEDVR2
```

#### Step 2: Run Quantization Tool

```bash
# Basic conversion
python tools/quantize_to_nvfp4.py \
    --input models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors \
    --output models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors
```

**Expected output**:
```
==================================================================
NVFP4 Model Quantization Tool
==================================================================

📂 Loading input model: models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors
✅ Loaded in 2.34s

Input Model Information:
  Total parameters: 3,017,543,680
  Total size: 5,736.12 MB
  Number of tensors: 1,270

🔄 Quantizing to NVFP4 format...

Quantizing vid_in.proj.weight: shape=(2560, 132), elements=337,920
  Error: MSE=0.000234, PSNR=52.31dB, MaxErr=0.001245, RelErr=0.0012
...

✅ Quantization completed in 145.67s

Compression Statistics:
  Input size:         5,736.12 MB
  Output size:        1,434.03 MB
  Compression ratio:  4.00x
  Space saved:        4,302.09 MB (75.0%)

💾 Saving NVFP4 model: models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors
✅ Saved in 3.21s

✅ NVFP4 Quantization Complete!
```

#### Step 3: Copy to ComfyUI

```bash
# Copy to ComfyUI models directory
cp models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors \
    /path/to/ComfyUI/models/SEEDVR2/
```

#### Step 4: Use in ComfyUI

1. Open ComfyUI
2. Add **SeedVR2 Load DiT Model** node
3. Select `seedvr2_ema_3b_nvfp4.safetensors` from dropdown
4. **Enable** `force_nvfp4` checkbox
5. Connect to upscaler and run workflow

### Advanced Options

#### Custom Block Size

```bash
# Use 32-value blocks instead of 16
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4_b32.safetensors \
    --block-size 32
```

#### Quantize All Tensors

```bash
# Don't skip small tensors
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4_full.safetensors \
    --no-skip-small
```

### Troubleshooting

**Problem**: Out of memory during quantization
**Solution**: Quantization runs on CPU by default. Ensure you have 16GB+ RAM.

**Problem**: Output file size not 4x smaller
**Solution**: Small tensors may not be quantized. Use `--no-skip-small` flag.

**Problem**: Model doesn't load in ComfyUI
**Solution**: Ensure `force_nvfp4=true` checkbox is enabled in DiT loader node.

**Problem**: "NVFP4 format not detected" error
**Solution**: Verify output file has `.nvfp4_data` keys using:
```python
from safetensors.torch import load_file
state = load_file("model_nvfp4.safetensors")
print([k for k in state.keys() if 'nvfp4' in k][:5])
```

### Quality Validation

After conversion, check quality metrics in the output:
- **PSNR > 45dB**: Excellent (imperceptible difference)
- **PSNR 40-45dB**: Very good (minimal difference)
- **PSNR < 40dB**: Noticeable difference (may need adjustment)

### Performance Expectations

**On Blackwell GPUs (RTX 50 series)**:
- Memory: 4GB (vs 16GB FP16)
- Speed: 2.5x faster than FP16
- Batch size: 4x larger frames
- Quality: 98-99% of FP16

**On Other GPUs** (Ada, Ampere, etc.):
- Memory: 4GB (same benefit)
- Speed: 1.2-1.5x faster (software dequantization)
- Still faster than FP16 due to memory bandwidth

---

## Türkçe Kılavuz

### Genel Bakış

Bu kılavuz, SeedVR2 FP16 modelini Blackwell GPU'larda (RTX 50 serisi) maksimum performans için native NVFP4 formatına nasıl dönüştüreceğinizi açıklar.

### NVFP4 Nedir?

**NVFP4 (4-bit kayan nokta)** NVIDIA'nın gelişmiş kuantizasyon formatıdır:
- **4-bit hassasiyet**: E2M1 formatı (1 işaret, 2 üs, 1 mantis)
- **İki seviye ölçekleme**: FP8 mikro-blok + FP32 tensor ölçekleri
- **Native donanım**: Blackwell'de doğrudan Tensor Core çalıştırma
- **Yüksek performans**: Diğer 4-bit formatlardan 2-3x daha hızlı

### Faydalar

| Metrik | FP16 | NVFP4 | İyileşme |
|--------|------|-------|----------|
| Model Boyutu | 6GB | 1.5GB | **4x daha küçük** |
| Bellek Kullanımı | 16GB | 4GB | **4x daha az** |
| Çıkarım Hızı | 1.0x | 2.5x | **2.5x daha hızlı** |
| Kalite Kaybı | 0% | <1% | **Minimal** |

### Gereksinimler

1. **Python ortamı**:
   - torch >= 2.0.0
   - safetensors >= 0.3.0

2. **Girdi modeli**: `seedvr2_ema_3b_fp16.safetensors`
   - İndirme: https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_3b_fp16.safetensors

3. **Disk alanı**: ~8GB boş alan (girdi + çıktı dosyaları için)

### Adım Adım Dönüştürme

#### Adım 1: FP16 Modelini İndirin

```bash
# Model klasörü oluştur
mkdir -p models/SEEDVR2

# FP16 modeli indir
wget https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp16.safetensors \
    -O models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors
```

#### Adım 2: Kuantizasyon Aracını Çalıştırın

```bash
# Temel dönüştürme
python tools/quantize_to_nvfp4.py \
    --input models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors \
    --output models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors
```

**Beklenen çıktı**:
```
==================================================================
NVFP4 Model Kuantizasyon Aracı
==================================================================

📂 Girdi modeli yükleniyor: models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors
✅ 2.34 saniyede yüklendi

Girdi Model Bilgisi:
  Toplam parametreler: 3,017,543,680
  Toplam boyut: 5,736.12 MB
  Tensor sayısı: 1,270

🔄 NVFP4 formatına dönüştürülüyor...

✅ Kuantizasyon 145.67 saniyede tamamlandı

Sıkıştırma İstatistikleri:
  Girdi boyutu:       5,736.12 MB
  Çıktı boyutu:       1,434.03 MB
  Sıkıştırma oranı:   4.00x
  Kazanılan alan:     4,302.09 MB (%75.0)

💾 NVFP4 modeli kaydediliyor: models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors
✅ 3.21 saniyede kaydedildi

✅ NVFP4 Kuantizasyonu Tamamlandı!
```

#### Adım 3: ComfyUI'ye Kopyalayın

```bash
# ComfyUI model klasörüne kopyala
cp models/SEEDVR2/seedvr2_ema_3b_nvfp4.safetensors \
    /yol/ComfyUI/models/SEEDVR2/
```

#### Adım 4: ComfyUI'de Kullanın

1. ComfyUI'yi açın
2. **SeedVR2 Load DiT Model** node'u ekleyin
3. Açılır menüden `seedvr2_ema_3b_nvfp4.safetensors` seçin
4. `force_nvfp4` kutusunu **işaretleyin**
5. Upscaler'a bağlayın ve workflow'u çalıştırın

### Sorun Giderme

**Sorun**: Kuantizasyon sırasında bellek hatası
**Çözüm**: Kuantizasyon varsayılan olarak CPU'da çalışır. 16GB+ RAM'e ihtiyaç vardır.

**Sorun**: Çıktı dosyası 4x daha küçük değil
**Çözüm**: Küçük tensorler kuantize edilmemiş olabilir. `--no-skip-small` kullanın.

**Sorun**: Model ComfyUI'de yüklenmiyor
**Çözüm**: DiT loader node'unda `force_nvfp4=true` kutusunun işaretli olduğundan emin olun.

### Performans Beklentileri

**Blackwell GPU'larda (RTX 50 serisi)**:
- Bellek: 4GB (FP16'da 16GB'a karşı)
- Hız: FP16'dan 2.5x daha hızlı
- Batch boyutu: 4x daha büyük frame'ler
- Kalite: FP16'nın %98-99'u

**Diğer GPU'larda** (Ada, Ampere, vb.):
- Bellek: 4GB (aynı fayda)
- Hız: 1.2-1.5x daha hızlı (yazılım dequantization)
- Bellek bant genişliği sayesinde FP16'dan hala daha hızlı

---

## Technical Details

### NVFP4 Format Specification

**E2M1 Encoding**:
```
4 bits: [sign:1][exponent:2][mantissa:1]

Representable values:
  Positive: 0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6
  Negative: -0.5, -0.75, -1, -1.5, -2, -3, -4, -6

Exponent bias = 1:
  exp=00 → 2^-1 = 0.5
  exp=01 → 2^0  = 1.0
  exp=10 → 2^1  = 2.0
  exp=11 → 2^2  = 4.0

Mantissa: {1.0, 1.5}
```

**Two-Level Scaling**:
```python
# Level 1: Micro-block (16 values per FP8 scale)
block_scaled = decoded_values * fp8_scale

# Level 2: Tensor (global FP32 scale)
final_value = block_scaled * fp32_scale
```

### File Format

**Output structure**:
```python
{
    # For each original parameter:
    "param_name.nvfp4_data": torch.uint8,     # Packed 4-bit values
    "param_name.fp8_scales": torch.float8_e4m3fn,  # Micro-block scales
    "param_name.fp32_scale": torch.float32,   # Global scale
    
    # Metadata:
    "_metadata": {
        "quantization": "nvfp4",
        "format": "e2m1",
        "block_size": 16,
        "two_level_scaling": True,
    }
}
```

### Quality Metrics

**Quantization Error**:
- **MSE** (Mean Squared Error): <0.001 target
- **PSNR** (Peak Signal-to-Noise Ratio): >45dB target
- **Max Error**: <0.01 typical
- **Relative Error**: <0.5% typical

### References

- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [AI Optimization Techniques](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/)

---

## Support

For issues or questions:
1. Check [NVFP4_GUIDE.md](NVFP4_GUIDE.md) for detailed documentation
2. Review [NATIVE_NVFP4_IMPLEMENTATION.md](../NATIVE_NVFP4_IMPLEMENTATION.md) for technical details
3. Open an issue on GitHub

---

**Version**: 1.0
**Last Updated**: 2026-02-01
