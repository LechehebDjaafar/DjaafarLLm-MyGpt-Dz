# ✅ هذا الملف الوحيد الذي تعدّله
DATA_CONFIG = {
    "file_path": "data/data.txt",   # ← غيّر المسار فقط هنا
}

MODEL_CONFIG = {
    "seq_len": 64,
    "vocab_size": None,       # يُحدَّد تلقائياً بعد Tokenizer
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "lr": 3e-4,
    "save_path": "model.pt",
}