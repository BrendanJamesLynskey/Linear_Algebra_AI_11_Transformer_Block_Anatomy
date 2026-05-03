# Transformer Block Anatomy

Deck 11 of the [Linear Algebra for AI / ML](https://github.com/BrendanJamesLynskey/LLM_Hub_Linear_Algebra) series.

**Live presentation:** https://brendanjameslynskey.github.io/Linear_Algebra_AI_11_Transformer_Block_Anatomy/

Walk a tensor through a full pre-norm transformer block. Pre-norm, multi-head attention with all four projections, residual, FFN with up-then-down, residual. Every shape, every matmul, every parameter, plus a 40-line reference PyTorch implementation that compiles down to the structure inside Llama, Mistral and Qwen.

## What's inside

- The pre-norm block in one equation; identity start, gradient highway
- The residual stream as a linear bus; shared coordinate system, bandwidth budget, decomposability
- LayerNorm vs RMSNorm: the geometry of projecting onto the radius-$\sqrt{d}$ sphere
- Walking the tensor: 17 numbered steps from input to output, with shapes and FLOPs
- Parameter budget per block: $4d^2 + 3 d \cdot d_{ff}$, attention vs FFN ratio
- FLOP budget per token: the famous "$6N$" Chinchilla rule explained
- Where compute lives: attention as cross-position bandwidth, FFN as per-position memory
- Reference PyTorch implementation (~40 lines) with RMSNorm + GQA + RoPE + SwiGLU
- Modern variants: pre/post norm, GeLU/SwiGLU, bias-free, RoPE, GQA, MLA &mdash; what changed and what didn't

Single-page HTML, KaTeX-rendered maths, no build step. Open `index.html` directly.
