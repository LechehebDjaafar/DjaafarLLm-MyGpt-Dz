from flask import Flask, render_template, request, jsonify
from tokenizers import Tokenizer
from model import MiniLLM
from rag import RAG
import torch
import os
import re

app = Flask(__name__)

# ─── تحميل النماذج ───
MODELS_DIR = "models"
loaded_models = {}

def load_model(version):
    if version in loaded_models:
        return loaded_models[version]

    model_path     = os.path.join(MODELS_DIR, f"model{version}.pt")
    tokenizer_path = os.path.join(MODELS_DIR, f"my_tokenizer{version}.json")

    if not os.path.exists(model_path):
        return None, None

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
    loaded_models[version] = (model, tokenizer)
    print(f"✅ تم تحميل model{version}")
    return model, tokenizer

def generate(model, tokenizer, prompt, max_tokens=60, temperature=0.6, top_k=15):
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

    result = tokenizer.decode(clean).strip()
    for sep in ["\nمستخدم:", "\nUser:", "\nس:"]:
        if sep in result:
            result = result.split(sep)[0].strip()
            break
    result = re.sub(r'\s([?.،!])', r'\1', result)
    return result if result else "..."

# تحميل RAG
rag = RAG("data/data.txt")

# تحميل النماذج عند البدء
load_model("v1")
load_model("v2")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json
    message = data.get("message", "").strip()
    version = data.get("model", "v1")

    if not message:
        return jsonify({"response": "..."})

    # RAG أولاً
    rag_answer = rag.answer(message)
    if rag_answer:
        return jsonify({"response": rag_answer, "source": "RAG"})

    # إذا RAG ما لقى إجابة → النموذج
    model, tokenizer = load_model(version)
    if model is None:
        return jsonify({"response": f"❌ model{version} غير موجود"})

    response = generate(model, tokenizer, message)
    return jsonify({"response": response, "source": f"model{version}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)