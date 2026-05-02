import torch
from tokenizers import Tokenizer
from model import MiniLLM
import sys

def load_model(model_path="modelv2.pt", tokenizer_path="my_tokenizerv2.json"):
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

def generate_response(model, tokenizer, prompt, max_tokens=60, temperature=0.6, top_k=15):
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    formatted = f"مستخدم: {prompt.strip()}\nجعفر:"

    input_ids = tokenizer.encode(formatted).ids
    idx = torch.tensor([[bos_id] + input_ids], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens,
                                 temperature=temperature, top_k=top_k)

    generated    = output[0].tolist()
    response_ids = generated[len(input_ids) + 1:]

    clean = []
    for tid in response_ids:
        if tid == eos_id:
            break
        clean.append(tid)

    # فك التشفير كاملاً أولاً ثم نقطع نصياً
    result = tokenizer.decode(clean).strip()

    # قطع عند أي بداية سؤال جديد — نصياً وليس token بtoken
    stop_phrases = ["مستخدم:", "مستخدم :", "User:", "\nس:", "\nQ:"]
    for phrase in stop_phrases:
        if phrase in result:
            result = result.split(phrase)[0].strip()
            break

    # تنظيف المسافات الزائدة حول علامات الترقيم
    import re
    result = re.sub(r'\s([?.،!])', r'\1', result)

    return result if result else "..."
def chat():
    print("=" * 50)
    print("   🤖 مرحباً! أنا نموذج جعفر الشخصي")
    print("   اكتب سؤالك — اكتب (خروج) للإنهاء")
    print("=" * 50)

    print("\n⏳ جاري تحميل النموذج...")
    try:
        model, tokenizer = load_model("modelv2.pt", "my_tokenizerv2.json")
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
        print(f"🤖 جعفر: {response}\n")

if __name__ == "__main__":
    chat()