# ✅ هذا الملف الوحيد الذي تعدّله
DATA_CONFIG = {
    "file_path": "data/data.txt",
}

MODEL_CONFIG = {
    "seq_len": 128,        # ← من 64 إلى 128 (يفهم جمل أطول)
    "vocab_size": None,
    "embed_dim": 256,      # ← من 128 إلى 256 (نموذج أذكى)
    "num_heads": 4,
    "num_layers": 4,       # ← من 2 إلى 4 (تفكير أعمق)
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,          # ← من 10 إلى 50
    "lr": 3e-4,
    "save_path": "model.pt",
}