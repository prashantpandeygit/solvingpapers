import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, random
import tiktoken
import os
import pickle
import math

# Configuration
class Config:
    vocab_size = 50257  # Default for gpt2 encoding
    dim = 256
    n_layers = 6
    n_heads = 8
    n_kv_heads = 4
    max_seq_len = 512
    batch_size = 32
    learning_rate = 3e-4
    dropout_rate = 0.0

def rms_norm(x, weight, eps=1e-6):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * weight * jnp.reciprocal(jnp.sqrt(var + eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.complex64(jnp.exp(1j * freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])
    # xq_complex: (B, T, n_heads, head_dim//2)
    # freqs_cis: (T, head_dim//2)
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq, xk

def repeat_kv(x, n_rep):
    return x if n_rep == 1 else jnp.repeat(x, n_rep, axis=2)

def init_weight(key, shape, scale=None):
    scale = 1.0 / math.sqrt(shape[0]) if scale is None else scale
    return random.normal(key, shape) * scale

def init_attention_weights(key, dim, n_heads, n_kv_heads):
    keys = random.split(key, 4)
    head_dim = dim // n_heads
    return {
        'wq': init_weight(keys[0], (dim, n_heads * head_dim)),
        'wk': init_weight(keys[1], (dim, n_kv_heads * head_dim)),
        'wv': init_weight(keys[2], (dim, n_kv_heads * head_dim)),
        'wo': init_weight(keys[3], (n_heads * head_dim, dim))
    }

def init_ffn_weights(key, dim):
    keys = random.split(key, 3)
    # LLaMA typically uses a different hidden_dim calculation, but we keep 4*dim for now as in original
    hidden_dim = 4 * dim 
    return {
        'w1': init_weight(keys[0], (dim, hidden_dim)),
        'w2': init_weight(keys[1], (hidden_dim, dim)),
        'w3': init_weight(keys[2], (dim, hidden_dim))
    }

def init_transformer_block(key, dim, n_heads, n_kv_heads):
    keys = random.split(key, 4)
    return {
        'attention': init_attention_weights(keys[0], dim, n_heads, n_kv_heads),
        'ffn': init_ffn_weights(keys[1], dim),
        'attention_norm': init_weight(keys[2], (dim,), scale=1.0),
        'ffn_norm': init_weight(keys[3], (dim,), scale=1.0)
    }

def init_model_params(key, vocab_size, dim, n_layers, n_heads, n_kv_heads):
    keys = random.split(key, 4)
    params = {
        'token_embedding': init_weight(keys[0], (vocab_size, dim)),
        'norm_f': init_weight(keys[1], (dim,), scale=1.0),
        'output': init_weight(keys[2], (dim, vocab_size))
    }
    block_keys = random.split(keys[3], n_layers)
    params['blocks'] = [init_transformer_block(k, dim, n_heads, n_kv_heads) for k in block_keys]
    return params

def attention(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0):
    B, T, C = x.shape
    head_dim = C // n_heads
    q = jnp.dot(x, params['wq']).reshape(B, T, n_heads, head_dim)
    k = jnp.dot(x, params['wk']).reshape(B, T, n_kv_heads, head_dim)
    v = jnp.dot(x, params['wv']).reshape(B, T, n_kv_heads, head_dim)
    
    q, k = apply_rotary_emb(q, k, freqs_cis[position:position + T])
    
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=1)
        v = jnp.concatenate([cache[1], v], axis=1)
    new_cache = (k, v)
    
    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)
    
    q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    
    if mask is not None:
        # Mask should be (1, 1, T_query, T_key) or broadcastable
        scores = scores + mask[:, :, :T, :k.shape[2]]
        
    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return jnp.dot(output, params['wo']), new_cache

def feed_forward(params, x):
    # SwiGLU: (SiLU(xW1) * xW3) W2
    # The original notebook had (SiLU(xW3) * xW1) W2, we'll keep that but with fix if needed.
    return jnp.dot(jax.nn.silu(jnp.dot(x, params['w3'])) * jnp.dot(x, params['w1']), params['w2'])

def transformer_block(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0, training=False, dropout_rate=0.0, key=None):
    attn_output, new_cache = attention(params['attention'], rms_norm(x, params['attention_norm']), mask, freqs_cis, n_heads, n_kv_heads, cache, position)
    
    if training and dropout_rate > 0:
        dropout_key, key = random.split(key)
        attn_output = random.bernoulli(dropout_key, 1-dropout_rate, shape=attn_output.shape) * attn_output / (1-dropout_rate)
        
    h = x + attn_output
    ffn_output = feed_forward(params['ffn'], rms_norm(h, params['ffn_norm']))
    
    if training and dropout_rate > 0:
        dropout_key, key = random.split(key)
        ffn_output = random.bernoulli(dropout_key, 1-dropout_rate, shape=ffn_output.shape) * ffn_output / (1-dropout_rate)
        
    out = h + ffn_output
    return out, new_cache

def model_forward(params, inputs, config, cache=None, position=0):
    B, T = inputs.shape
    h = params['token_embedding'][inputs]
    
    # In a real implementation, freqs_cis and mask might be passed in for performance
    freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len)
    mask = jnp.tril(jnp.ones((config.max_seq_len, config.max_seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask.astype(h.dtype)
    mask = mask[None, None, :, :]
    
    new_caches = []
    for i, block in enumerate(params['blocks']):
        layer_cache = cache[i] if cache is not None else None
        # training=False here as per original
        h, layer_cache = transformer_block(block, h, mask, freqs_cis, config.n_heads, config.n_kv_heads, layer_cache, position, training=False, dropout_rate=config.dropout_rate)
        new_caches.append(layer_cache)
        
    h = rms_norm(h, params['norm_f'])
    logits = jnp.dot(h, params['output'])
    return logits, new_caches

def compute_loss(params, batch, config):
    inputs, targets = batch
    logits, _ = model_forward(params, inputs, config)
    logits = logits.reshape(-1, config.vocab_size)
    targets = targets.reshape(-1)
    loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            targets[:, None],
            axis=1
        )
    )
    return loss

@jax.jit
def update_step(params, batch, config):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch, config)
    params = jax.tree.map(
        lambda p, g: p - config.learning_rate * g,
        params,
        grads
    )
    return params, loss

def get_batch(key, data, batch_size, seq_len):
    ix = random.randint(key, (batch_size,), 0, len(data) - seq_len)
    x = vmap(lambda i: lax.dynamic_slice(data, (i,), (seq_len,)))(ix)
    y = vmap(lambda i: lax.dynamic_slice(data, (i + 1,), (seq_len,)))(ix)
    return x, y

def generate(params, prompt_tokens, max_new_tokens, config, key):
    x = jnp.array(prompt_tokens)
    for _ in range(max_new_tokens):
        x_crop = x[-config.max_seq_len:]
        logits, _ = model_forward(params, x_crop[None, :], config)
        logits = logits[0, -1, :]  # take the last logit
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, logits, shape=(1,))[0]
        x = jnp.concatenate([x, jnp.array([next_token])])
    return x.tolist()

if __name__ == "__main__":
    # Load data
    enc = tiktoken.get_encoding("gpt2")
    if os.path.exists('shakespeare.txt'):
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        data = jnp.array(tokens)
        print(f"Loaded {len(data)} tokens from shakespeare.txt")
    else:
        print("shakespeare.txt not found, using dummy data.")
        data = jnp.zeros((10000,), dtype=jnp.int32)

    config = Config()
    config.vocab_size = enc.n_vocab
    
    key = random.PRNGKey(0)
    params = init_model_params(
        key=key,
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads
    )
    
    print("Model initialized.")

    # Training loop
    num_epochs = 1
    steps_per_epoch = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            key, batch_key = random.split(key)
            batch = get_batch(batch_key, data, config.batch_size, config.max_seq_len)
            params, loss = update_step(params, batch, config)
            epoch_loss += loss
            if step % 2 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss:.4f}")
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / steps_per_epoch:.4f}")

    # Generation test
    print("\nGenerating text...")
    prompt = tokens[:10] if 'tokens' in locals() else [1, 2, 3]
    key, gen_key = random.split(key)
    output_tokens = generate(params, prompt, 20, config, gen_key)
    print(f"Generated tokens: {output_tokens}")
    print(f"Decoded: {enc.decode(output_tokens)}")
