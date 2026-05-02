import torch
from tokenizers import Tokenizer
from model import MiniLLM
from config import MODEL_CONFIG
import sys

def load_model(model_path="model.pt", tokenizer_path="my_tokenizer.json"):
    checkpoint = torch.load(model_path, map_location="cpu")
    tokenizer  = Tokenizer.from_file(tokenizer_path)
    cfg        = checkpoint["model_config"]

    model = MiniLLM(
        vocab_size = tokenizer.get_vocab_size(),
        embed_dim  = cfg["embed_dim"],
        num_heads  = cfg["num_heads"],
        num_layers = cfg["num_layers"],
        seq_len    = cfg["seq_len"],
        dropout    = 0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_tokens=80, temperature=0.7, top_k=30):
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([[bos_id] + input_ids], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens,
                                 temperature=temperature, top_k=top_k)

    generated = output[0].tolist()

    # إزالة prompt الأصلي و tokens خاصة
    response_ids = generated[len(input_ids) + 1:]
    if eos_id in response_ids:
        response_ids = response_ids[:response_ids.index(eos_id)]

    return tokenizer.decode(response_ids).strip()

def chat():
    print("=" * 50)
    print("   🤖 مرحباً! أنا نموذجك الشخصي")
    print("   اكتب سؤالك — اكتب (خروج) للإنهاء")
    print("=" * 50)

    print("\n⏳ جاري تحميل النموذج...")
    try:
        model, tokenizer = load_model("model.pt", "my_tokenizer.json")
        print("✅ النموذج جاهز!\n")
    except FileNotFoundError:
        print("❌ لم يُعثر على model.pt — شغّل train.py أولاً")
        sys.exit(1)

    while True:
        try:
            user_input = input("أنت: ").strip()
        except KeyboardInterrupt:
            print("\n👋 وداعاً!")
            break

        if not user_input:
            continue
        if user_input in ["خروج", "exit", "quit"]:
            print("👋 وداعاً!")
            break

        response = generate_response(model, tokenizer, user_input)

        if not response:
            response = "..."

        print(f"🤖 جعفر: {response}\n")

if __name__ == "__main__":
    chat()