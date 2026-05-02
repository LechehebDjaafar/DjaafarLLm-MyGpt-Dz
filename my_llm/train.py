import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import MiniLLM
from tokenizer_train import train_tokenizer
from data_loader import load_data
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
import time

# ─── Dataset ───
class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len
    def __len__(self):
        return len(self.tokens) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx     : idx + self.seq_len])
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return x, y

def save_checkpoint(model, tokenizer_path, epoch, loss, path="model.pt"):
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "loss":         loss,
        "model_config": MODEL_CONFIG,
        "tokenizer":    tokenizer_path,
    }, path)
    print(f"💾 محفوظ: {path} | epoch {epoch} | loss {loss:.4f}")

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  الجهاز: {device.upper()}")

    text      = load_data(DATA_CONFIG["file_path"])
    tokenizer = train_tokenizer(text, vocab_size=3000, save_path="my_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    MODEL_CONFIG["vocab_size"] = vocab_size

    all_tokens = tokenizer.encode(text).ids
    print(f"📊 إجمالي tokens: {len(all_tokens):,}")

    if len(all_tokens) < TRAIN_CONFIG["batch_size"] + MODEL_CONFIG["seq_len"]:
        print("⚠️  البيانات قليلة جداً — أضف نصوصاً أكثر في data.txt")
        return

    dataset    = TextDataset(all_tokens, MODEL_CONFIG["seq_len"])
    dataloader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, drop_last=True)
    print(f"📦 عدد batches: {len(dataloader)}")

    model = MiniLLM(
        vocab_size = vocab_size,
        embed_dim  = MODEL_CONFIG["embed_dim"],
        num_heads  = MODEL_CONFIG["num_heads"],
        num_layers = MODEL_CONFIG["num_layers"],
        seq_len    = MODEL_CONFIG["seq_len"],
        dropout    = MODEL_CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 معاملات النموذج: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["epochs"]
    )

    best_loss = float("inf")
    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0
        start = time.time()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        elapsed  = time.time() - start
        print(f"Epoch {epoch:02d}/{TRAIN_CONFIG['epochs']} | loss: {avg_loss:.4f} | ⏱ {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, "my_tokenizer.json", epoch, avg_loss, TRAIN_CONFIG["save_path"])

    print(f"\n✅ التدريب انتهى! أفضل loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()