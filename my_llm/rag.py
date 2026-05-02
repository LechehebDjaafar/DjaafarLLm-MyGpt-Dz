import json
import re
import math
from collections import Counter

def tokenize(text):
    """تقطيع النص لكلمات بسيطة"""
    return re.findall(r'\w+', text.lower())

def build_index(data_path):
    """يبني فهرس من data.txt"""
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    # فصل أزواج مستخدم/جعفر
    pairs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("مستخدم:") and i + 1 < len(lines):
            q = lines[i].replace("مستخدم:", "").strip()
            a = lines[i+1].replace("جعفر:", "").strip()
            pairs.append({"question": q, "answer": a})
            i += 2
        else:
            pairs.append({"question": lines[i], "answer": lines[i]})
            i += 1
    return pairs

def similarity(q1, q2):
    """حساب التشابه بين سؤالين"""
    w1 = Counter(tokenize(q1))
    w2 = Counter(tokenize(q2))
    common = set(w1) & set(w2)
    if not common:
        return 0.0
    num = sum(w1[c] * w2[c] for c in common)
    den = math.sqrt(sum(v**2 for v in w1.values())) * \
          math.sqrt(sum(v**2 for v in w2.values()))
    return num / den if den else 0.0

def retrieve(query, index, top_k=3):
    """يسترجع أقرب إجابات للسؤال"""
    scored = [(similarity(query, p["question"]), p) for p in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p for score, p in scored[:top_k] if score > 0.1]
    return results

class RAG:
    def __init__(self, data_path):
        self.index = build_index(data_path)
        print(f"✅ RAG جاهز | {len(self.index)} سجل مفهرس")

    def answer(self, query):
        """يرجع أفضل إجابة مباشرة من البيانات"""
        results = retrieve(query, self.index)
        if results:
            return results[0]["answer"]
        return None