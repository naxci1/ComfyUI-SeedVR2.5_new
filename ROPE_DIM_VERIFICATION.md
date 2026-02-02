# RoPE Dimension Verification

## Current Code State (Commit 3f30778)

### File 1: src/models/dit_3b/nadit.py (Lines 84-88)

```python
# FORCED for NVFP4 3B model
if head_dim == 64:
    rope_dim = 64  # Use full head_dim to ensure proper freqs initialization
else:
    rope_dim = rope_dim if rope_dim is not None else head_dim // 2
```

### File 2: src/models/dit_3b/rope.py (Lines 28-30)

```python
self.rope = RotaryEmbedding(
    # FORCED for NVFP4 3B: Ensure freqs=[21] to match checkpoint
    dim=21 if (dim == 64 and rope_dim >= 60) else dim // rope_dim,
```

## Expected Behavior

### Calculation Flow

1. **In nadit.py**:
   - head_dim = 64 (from 1280 / 20 heads)
   - Condition: head_dim == 64 → TRUE
   - Action: rope_dim = 64

2. **In rope.py RotaryEmbeddingBase.__init__**:
   - Receives: dim=64, rope_dim=64
   - Condition: (64 == 64 and 64 >= 60) → TRUE
   - Action: dim = 21 (forced)

3. **In RotaryEmbedding**:
   - Receives: dim=21
   - Creates: freqs of Size([21])

### Result

✓ freqs = Size([21]) matching checkpoint
✗ NOT Size([10]) which would come from rope_dim=60

## Verification Commands

```bash
# Check nadit.py
sed -n '84,88p' src/models/dit_3b/nadit.py

# Check rope.py
sed -n '28,30p' src/models/dit_3b/rope.py

# Verify commit
git log --oneline -1
# Should show: 3f30778 REAL FIX: rope_dim=64 + force dim=21
```

## If Still Seeing Size([10])

Possible causes:
1. **Old cached model** - Clear ComfyUI model cache
2. **Different checkpoint** - Verify using seedvr2_ema_3b_nvfp4_native.safetensors
3. **Different execution path** - Check if code is using different branch
4. **Python bytecode cache** - Delete __pycache__ directories and .pyc files

## Commit History

- 3f30778: rope_dim=64 + force dim=21 (CURRENT)
- 3891308: rope_dim=3 (caused Size([0])) - REVERTED
- 477879f: rope_dim=60 (caused Size([10])) - REVERTED

Current code should produce Size([21]) ✓
