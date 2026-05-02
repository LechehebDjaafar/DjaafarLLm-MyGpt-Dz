from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

def train_tokenizer(text: str, vocab_size: int = 3000, save_path: str = "my_tokenizer.json"):
    tmp_path = "tmp_train.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    tokenizer.train(files=[tmp_path], trainer=trainer)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )

    tokenizer.save(save_path)
    os.remove(tmp_path)

    print(f"✅ Tokenizer محفوظ: {save_path}")
    print(f"📦 حجم المفردات: {tokenizer.get_vocab_size()}")
    return tokenizer

def load_tokenizer(path: str = "my_tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)

def encode(tokenizer, text: str):
    return tokenizer.encode(text).ids

def decode(tokenizer, ids: list):
    return tokenizer.decode(ids)

# ── تشغيل مباشر ──
if __name__ == "__main__":
    from data_loader import load_data
    from config import DATA_CONFIG

    text = load_data(DATA_CONFIG["file_path"])
    tokenizer = train_tokenizer(text, vocab_size=3000, save_path="my_tokenizer.json")

    test = "أنا أحب البرمجة"
    ids = encode(tokenizer, test)
    back = decode(tokenizer, ids)
    print(f"\n🧪 اختبار:")
    print(f"  النص الأصلي : {test}")
    print(f"  الأرقام     : {ids}")
    print(f"  العودة      : {back}")