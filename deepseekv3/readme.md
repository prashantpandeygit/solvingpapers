## DeepSeekV3 8x2 Mixture of Experts Model
weights: https://huggingface.co/parkneurals/DeepSeekV3-8x2MoE

## Architecture

| Model Type | Decoder (Transformer)|
| Layers | 6 |
| Hidden Size | 512 |
| Attention Heads | 8 |
| Context Length | 256 |
| Vocabulary Size | 50257 (GPT-2 BPE) |
| Base Frequency | 100000 |
| Latent Dim | 64 |
| MoE | 8 experts, top-2 routing |
| MTP Heads | 0 (disabled) |

## Configuration

| Parameter | Value |
|----------|------|
| Dataset | TinyStories |
| Epochs | 1 |
| Batch Size | 16 |
| Max Learning Rate | 6e-4 |
| Gradient Clipping | 1.0 |

## Regularization

| Parameter | Value |
|----------|------|
| Attention Dropout | 0.1 |
| Dropout | 0.1 |

## Optimization

| Parameter | Value |
|----------|------|
| Optimizer | AdamW |
| Weight Decay | 0.1 |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| Epsilon | 1e-8 |
| Loss Scale | 0.03 |

## Mixture-of-Experts (MoE)

| Parameter | Value |
|----------|------|
| Experts | 8 |
| Top-k Experts | 2 |
| Noisy Top-k | False |
| Shared Experts | Enabled |
| Aux-Free Load Balancing | Enabled |
| Bias Update Rate | 0.001 |

## Hardware

| Parameter | Value |
|----------|------|
| Platform | Kaggle |
| GPU | 2xTesla T4 |

## Training

| Metric | Value |
|-------|------|
| Final Loss | 2.90068 |
| Perplexity | 18.18644 |
| Steps | 10000 |
| Tokens Seen | 40960000 |
