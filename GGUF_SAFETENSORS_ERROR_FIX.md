# GGUF File Extension Error - Fix Guide

## Turkish / Türkçe

### Hata Mesajı
```
Error(s) in loading state_dict for NaDiT:
  size mismatch for vid_in.proj.weight: copying a param with shape torch.Size([21120, 16])
  from checkpoint, the shape in current model is torch.Size([2560, 132])
```

### Ne Oldu?
Bu hata, **GGUF formatındaki bir dosyanın yanlış uzantıyla (.safetensors) yüklenmeye çalışılmasından** kaynaklanıyor.

### Neden Oluyor?
1. Dosya adı: `seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors`
2. Ama içeriği: **GGUF Q4_K_M quantized data**
3. Şekil pattern'i: `[21120, 16]` → GGUF block quantization
4. Beklenen pattern: `[2560, 132]` → Normal safetensors

### ÇÖZÜM 1: Dosya Adını Değiştir (Önerilen)

**Adım 1: Dosyayı Bul**
```
Konum: D:\ComfyUI\ComfyUI\models\SEEDVR2\
Dosya: seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors
```

**Adım 2: Yeniden Adlandır**
```
Eski: seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors
Yeni: seedvr2_3b_blackwell_nvfp4_extreme_full.gguf
```

**Adım 3: ComfyUI'yi Yeniden Başlat**
- Dosya artık GGUF olarak algılanacak
- Doğru şekilde yüklenecek

### ÇÖZÜM 2: Alternatif Model Kullan

Eğer dosya hala çalışmazsa, test edilmiş alternatifler:

**4-bit Quantization (GGUF):**
```
Model: seedvr2_ema_3b-Q4_K_M.gguf
Kaynak: https://huggingface.co/cmeka/SeedVR2-GGUF
VRAM: ~6GB
GPU: Herhangi bir CUDA GPU
```

**8-bit FP8 (Hızlı ve Kaliteli):**
```
Model: seedvr2_ema_3b_fp8_e4m3fn.safetensors
VRAM: ~10GB
GPU: RTX 40/50 serisi
Durum: Varsayılan modellerde mevcut
```

**16-bit FP16 (Maksimum Kalite):**
```
Model: seedvr2_ema_3b_fp16.safetensors
VRAM: ~16GB
GPU: Herhangi bir GPU
Durum: Varsayılan modellerde mevcut
```

### Teknik Detaylar

**GGUF Pattern Algılama:**
- Tensörler `[..., 16]` şekliyle bitiyor
- Q4_K_M block quantization imzası
- 10 tensörden 5+ tanesi bu pattern'de → GGUF

**Yeni Koruma:**
Artık kod otomatik olarak algılıyor ve net hata mesajı veriyor:
```python
# src/core/model_loader.py içinde
_detect_gguf_in_safetensors(state, checkpoint_path, debug)
```

### Sık Sorulan Sorular

**S: NVFP4 model gerçek mi?**
C: Hayır, NVFP4 henüz PyTorch'ta mevcut değil. Bu dosya muhtemelen GGUF.

**S: Neden .safetensors uzantılı?**
C: Muhtemelen yanlış isimlendirilmiş veya dönüştürme hatası.

**S: Hangi modeli kullanmalıyım?**
C: GGUF Q4_K_M veya FP8 öneriyoruz (her ikisi de iyi çalışıyor).

---

## English

### Error Message
```
Error(s) in loading state_dict for NaDiT:
  size mismatch for vid_in.proj.weight: copying a param with shape torch.Size([21120, 16])
  from checkpoint, the shape in current model is torch.Size([2560, 132])
```

### What Happened?
This error occurs when a **GGUF format file with wrong extension (.safetensors) is being loaded**.

### Why Does This Happen?
1. Filename: `seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors`
2. But contains: **GGUF Q4_K_M quantized data**
3. Shape pattern: `[21120, 16]` → GGUF block quantization
4. Expected pattern: `[2560, 132]` → Normal safetensors

### SOLUTION 1: Rename the File (Recommended)

**Step 1: Locate the File**
```
Location: D:\ComfyUI\ComfyUI\models\SEEDVR2\
File: seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors
```

**Step 2: Rename**
```
Old: seedvr2_3b_blackwell_nvfp4_extreme_full.safetensors
New: seedvr2_3b_blackwell_nvfp4_extreme_full.gguf
```

**Step 3: Restart ComfyUI**
- File will now be detected as GGUF
- Will load correctly

### SOLUTION 2: Use Alternative Model

If file still doesn't work, use tested alternatives:

**4-bit Quantization (GGUF):**
```
Model: seedvr2_ema_3b-Q4_K_M.gguf
Source: https://huggingface.co/cmeka/SeedVR2-GGUF
VRAM: ~6GB
GPU: Any CUDA GPU
```

**8-bit FP8 (Fast & Quality):**
```
Model: seedvr2_ema_3b_fp8_e4m3fn.safetensors
VRAM: ~10GB
GPU: RTX 40/50 series
Status: Available in default models
```

**16-bit FP16 (Maximum Quality):**
```
Model: seedvr2_ema_3b_fp16.safetensors
VRAM: ~16GB
GPU: Any GPU
Status: Available in default models
```

### Technical Details

**GGUF Pattern Detection:**
- Tensors ending with `[..., 16]` shape
- Q4_K_M block quantization signature
- 5+ out of 10 tensors with this pattern → GGUF

**New Protection:**
Code now auto-detects and provides clear error message:
```python
# In src/core/model_loader.py
_detect_gguf_in_safetensors(state, checkpoint_path, debug)
```

### FAQ

**Q: Is NVFP4 model real?**
A: No, NVFP4 doesn't exist in PyTorch yet. This file is likely GGUF.

**Q: Why .safetensors extension?**
A: Probably misnamed or conversion error.

**Q: Which model should I use?**
A: We recommend GGUF Q4_K_M or FP8 (both work well).

## Error Detection Logic

The fix adds automatic detection before loading:

```python
def _detect_gguf_in_safetensors(state_dict, checkpoint_path, debug):
    """
    Check first 10 tensors for GGUF Q4_K_M pattern:
    - If 5+ tensors have shape [..., 16] → GGUF detected
    - Raise clear error with solution steps
    """
    shapes_ending_16 = 0
    for key, tensor in list(state_dict.items())[:10]:
        if len(tensor.shape) >= 2 and tensor.shape[-1] == 16:
            shapes_ending_16 += 1
    
    if shapes_ending_16 >= 5:
        raise ValueError("GGUF data in .safetensors file...")
```

## Resolution Steps

1. **Immediate Fix**: Rename file to `.gguf` extension
2. **Alternative**: Download proven GGUF/FP8 models
3. **Future**: True NVFP4 support when PyTorch adds it

## Prevention

- Always check file extension matches content
- GGUF files must have `.gguf` extension
- SafeTensors files contain unquantized or standard quantized tensors
- Model names with "nvfp4" are likely misleading if NVFP4 isn't standardized yet

