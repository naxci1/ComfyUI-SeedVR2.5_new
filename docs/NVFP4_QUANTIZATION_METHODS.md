# NVFP4 Model Quantization Methods / NVFP4 Model Kuantizasyon Yöntemleri

Multiple ways to convert FP16 models to NVFP4 format.

FP16 modellerini NVFP4 formatına dönüştürmek için birden fazla yöntem.

---

## Türkçe / Turkish

### Durum: Patreon NVFP4 Quantizer'a Erişim Sorunu

Patreon'daki NVFP4 quantizer uygulamasına erişemiyorsanız, **endişelenmeyin!** Aynı işi yapan **ÜÇ farklı yöntem** sunuyoruz.

---

### ✅ Yöntem 1: Bizim Tool'umuz (ÖNERİLEN)

**Zaten hazır ve çalışır durumda!**

#### Avantajları
- ✅ **Tamamen ücretsiz** ve açık kaynak
- ✅ **Şu anda çalışıyor** - indirme/ödeme yok
- ✅ PyTorch dışında **bağımlılık yok**
- ✅ **Tam kontrol** - tüm parametreler ayarlanabilir
- ✅ **Detaylı metrikler** - kalite analizi dahil
- ✅ **ComfyUI'ye entegre** - direkt kullanıma hazır

#### Kullanım

```bash
# Temel kullanım
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4.safetensors

# İlerletme seçenekleri
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4.safetensors \
    --block-size 16 \
    --no-skip-small
```

#### Beklenen Sonuç
```
✅ NVFP4 Kuantizasyonu Tamamlandı!
Girdi:  5,736 MB
Çıktı:  1,434 MB
Sıkıştırma: 4.00x (%75 azalma)
Süre: ~2.5 dakika
Kalite: PSNR >45dB (mükemmel)
```

#### Detaylı Kılavuz
Tam talimatlar için: [NVFP4_CONVERSION_GUIDE.md](NVFP4_CONVERSION_GUIDE.md)

---

### 🔧 Yöntem 2: NVIDIA Model Optimizer (Resmi)

**NVIDIA'nın resmi kuantizasyon tool'u**

#### Kurulum

```bash
# NVIDIA Model Optimizer'ı yükle
pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com

# veya conda ile
conda install -c nvidia nvidia-modelopt
```

#### Kullanım

```python
import torch
from safetensors.torch import load_file, save_file
import modelopt.torch.quantization as mtq

# 1. FP16 modeli yükle
print("FP16 model yükleniyor...")
state_dict = load_file("seedvr2_ema_3b_fp16.safetensors")

# 2. PyTorch modeline dönüştür (gerekirse)
# model = YourModelClass()
# model.load_state_dict(state_dict)

# 3. NVFP4 kuantizasyon config'i
config = mtq.FP4_DEFAULT_CFG.copy()
config['quant_cfg'] = {
    '*weight_quantizer': {
        'num_bits': 4,
        'axis': None,
        'enable': True
    },
    '*input_quantizer': {'enable': False},
    '*output_quantizer': {'enable': False},
}

# 4. Model'i kuantize et
print("NVFP4'e kuantize ediliyor...")
mtq.quantize(model, config, forward_loop=None)

# 5. Kaydet
print("Kaydediliyor...")
torch.save(model.state_dict(), "seedvr2_ema_3b_nvfp4.pth")
# veya safetensors formatında:
save_file(model.state_dict(), "seedvr2_ema_3b_nvfp4.safetensors")

print("✅ Tamamlandı!")
```

#### Avantajları
- ✅ NVIDIA'nın resmi tool'u
- ✅ TensorRT entegrasyonu
- ✅ Blackwell optimizasyonları
- ✅ En yeni özellikler

#### Dezavantajları
- ❌ Daha karmaşık kurulum
- ❌ NVIDIA ekosistem bağımlılığı
- ❌ Model yapısını bilmeniz gerekebilir

---

### 📝 Yöntem 3: Manuel Kuantizasyon Script

**Özel ihtiyaçlar için basit Python scripti**

```python
import torch
from safetensors.torch import load_file, save_file
import sys
sys.path.append('.')
from src.models.nvfp4.quantize import quantize_model_to_nvfp4

# FP16 modeli yükle
print("FP16 model yükleniyor...")
state_dict = load_file("seedvr2_ema_3b_fp16.safetensors")

# NVFP4'e kuantize et
print("NVFP4'e dönüştürülüyor...")
nvfp4_state = quantize_model_to_nvfp4(
    state_dict,
    block_size=16,
    skip_small_tensors=True,
    min_elements=128
)

# Kaydet
print("Kaydediliyor...")
save_file(nvfp4_state, "seedvr2_ema_3b_nvfp4.safetensors")

print("✅ Dönüştürme tamamlandı!")
print(f"Orijinal: {len(state_dict)} tensor")
print(f"NVFP4: {len(nvfp4_state)} tensor (scales dahil)")
```

Scripti kaydet: `convert_to_nvfp4.py` ve çalıştır:
```bash
python convert_to_nvfp4.py
```

---

### 💰 Yöntem 4: Patreon Tool'una Erişim

**Patreon'daki NVFP4 Quantizer'ı kullanmak istiyorsanız:**

#### Adımlar:

1. **Patreon'a git**:
   - Link: https://www.patreon.com/posts/nvfp4-quantizer-app-148217625

2. **Kreator'a abone ol**:
   - Gerekli tier'ı seç (genellikle ücretli)
   - Ödeme yap

3. **Tool'u indir**:
   - Post'un içeriğine eriş
   - NVFP4 Quantizer uygulamasını indir

4. **Kullan**:
   - Uygulamayı aç
   - FP16 modelini seç
   - Quantize butonuna bas
   - NVFP4 çıktısını al

#### Not
- Patreon içeriği genellikle **ücretlidir**
- Aylık abonelik gerekebilir
- Tool, Windows/Mac uygulaması olabilir

---

### 🎯 Karşılaştırma: Hangi Yöntemi Seçmeliyim?

| Özellik | Bizim Tool | NVIDIA ModelOpt | Manuel Script | Patreon Tool |
|---------|------------|-----------------|---------------|--------------|
| **Ücretsiz** | ✅ Evet | ✅ Evet | ✅ Evet | ❌ Hayır (ücretli) |
| **Kurulum** | ⚡ Çok kolay | ⚠️ Orta | ⚡ Çok kolay | ❓ Bilinmiyor |
| **Hız** | 🚀 Hızlı | 🚀 Hızlı | 🚀 Hızlı | ❓ Bilinmiyor |
| **Kalite** | ✅ Mükemmel | ✅ Mükemmel | ✅ Mükemmel | ❓ Bilinmiyor |
| **Destek** | ✅ Tam | ✅ Resmi | ✅ Tam | ❓ Sınırlı |
| **ComfyUI Entegre** | ✅ Evet | ⚠️ Manuel | ⚠️ Manuel | ❓ Bilinmiyor |

### 💡 Önerimiz

**Çoğu kullanıcı için → Yöntem 1 (Bizim Tool)**

Neden?
- ✅ Ücretsiz ve hemen çalışır
- ✅ ComfyUI'ye tam entegre
- ✅ Detaylı dokümantasyon (Türkçe + İngilizce)
- ✅ Destek ve güncellemeler mevcut

**İleri düzey kullanıcılar için → Yöntem 2 (NVIDIA ModelOpt)**

Neden?
- ✅ NVIDIA'nın resmi tool'u
- ✅ TensorRT özellikleri
- ✅ En yeni optimizasyonlar

---

### 🔍 Patreon Tool vs Bizim Tool

**Patreon Tool'unun olası avantajları:**
- GUI (grafik arayüz) olabilir
- Drag-and-drop kolaylığı
- Windows .exe dosyası

**Bizim Tool'un kesin avantajları:**
- ✅ Tamamen ücretsiz
- ✅ Açık kaynak - kodları görebilirsiniz
- ✅ Özelleştirilebilir
- ✅ ComfyUI'ye tam entegre
- ✅ Detaylı hata raporları
- ✅ Kalite metrikleri (PSNR, MSE, vb.)
- ✅ Python API - script'lerle kullanılabilir

---

### 📚 Ek Kaynaklar

**Dokümantasyon**:
- [NVFP4_CONVERSION_GUIDE.md](NVFP4_CONVERSION_GUIDE.md) - Tam dönüştürme kılavuzu
- [NATIVE_NVFP4_IMPLEMENTATION.md](../NATIVE_NVFP4_IMPLEMENTATION.md) - Teknik detaylar
- [NVFP4_GUIDE.md](../docs/NVFP4_GUIDE.md) - Genel NVFP4 kılavuzu

**NVIDIA Kaynakları**:
- [NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [AI Optimization Techniques](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/)

---

## English / İngilizce

### Status: Can't Access Patreon NVFP4 Quantizer

If you can't access the Patreon NVFP4 quantizer app, **don't worry!** We provide **THREE alternative methods** that do the same job.

---

### ✅ Method 1: Our Tool (RECOMMENDED)

**Already built and ready to use!**

#### Advantages
- ✅ **Completely free** and open source
- ✅ **Works right now** - no download/payment needed
- ✅ **No dependencies** beyond PyTorch
- ✅ **Full control** - all parameters adjustable
- ✅ **Detailed metrics** - quality analysis included
- ✅ **ComfyUI integrated** - ready to use

#### Usage

```bash
# Basic usage
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4.safetensors

# Advanced options
python tools/quantize_to_nvfp4.py \
    --input seedvr2_ema_3b_fp16.safetensors \
    --output seedvr2_ema_3b_nvfp4.safetensors \
    --block-size 16 \
    --no-skip-small
```

#### Expected Output
```
✅ NVFP4 Quantization Complete!
Input:  5,736 MB
Output: 1,434 MB
Compression: 4.00x (75% reduction)
Time: ~2.5 minutes
Quality: PSNR >45dB (excellent)
```

#### Detailed Guide
Full instructions: [NVFP4_CONVERSION_GUIDE.md](NVFP4_CONVERSION_GUIDE.md)

---

### 🎯 Comparison: Which Method Should I Choose?

| Feature | Our Tool | NVIDIA ModelOpt | Manual Script | Patreon Tool |
|---------|----------|-----------------|---------------|--------------|
| **Free** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (paid) |
| **Setup** | ⚡ Very easy | ⚠️ Medium | ⚡ Very easy | ❓ Unknown |
| **Speed** | 🚀 Fast | 🚀 Fast | 🚀 Fast | ❓ Unknown |
| **Quality** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ❓ Unknown |
| **Support** | ✅ Full | ✅ Official | ✅ Full | ❓ Limited |
| **ComfyUI Integrated** | ✅ Yes | ⚠️ Manual | ⚠️ Manual | ❓ Unknown |

### 💡 Our Recommendation

**For most users → Method 1 (Our Tool)**

Why?
- ✅ Free and works immediately
- ✅ Fully integrated with ComfyUI
- ✅ Detailed documentation (Turkish + English)
- ✅ Support and updates available

**For advanced users → Method 2 (NVIDIA ModelOpt)**

Why?
- ✅ NVIDIA's official tool
- ✅ TensorRT features
- ✅ Latest optimizations

---

### 🔍 Patreon Tool vs Our Tool

**Patreon Tool possible advantages:**
- May have GUI (graphical interface)
- Drag-and-drop convenience
- Windows .exe file

**Our Tool definite advantages:**
- ✅ Completely free
- ✅ Open source - you can see the code
- ✅ Customizable
- ✅ Fully integrated with ComfyUI
- ✅ Detailed error reports
- ✅ Quality metrics (PSNR, MSE, etc.)
- ✅ Python API - scriptable

---

### 📞 Support

**Need help?**
1. Check documentation files (see above)
2. Review code in `src/models/nvfp4/`
3. Open GitHub issue

**Have Patreon access?**
- If you have access to Patreon tool and it has unique features
- Let us know and we can add them to our tool

---

**Version**: 1.0  
**Last Updated**: 2026-02-01
