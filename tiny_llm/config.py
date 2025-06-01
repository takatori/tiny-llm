GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False,
}
GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,
    "emb_dim": 1024,
    "context_length": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False,
}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257,
    "emb_dim": 1280,
    "context_length": 1024,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False,    
}

GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "emb_dim": 1600,
    "context_length": 1024,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,
    "qkv_bias": False,    
}