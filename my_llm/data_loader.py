import os
import json
import csv

def detect_format(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    formats = {
        ".txt":   "txt",
        ".json":  "json",
        ".jsonl": "jsonl",
        ".csv":   "csv",
    }
    if ext not in formats:
        raise ValueError(f"❌ صيغة غير مدعومة: {ext} — المدعوم: {list(formats.keys())}")
    return formats[ext]

def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.extend([str(v) for v in item.values()])
        return " ".join(texts)
    return str(data)

def load_jsonl(file_path: str) -> str:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                texts.extend([str(v) for v in item.values()])
            else:
                texts.append(str(item))
    return " ".join(texts)

def load_csv(file_path: str) -> str:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            texts.extend(row)
    return " ".join(texts)

def load_data(file_path: str) -> str:
    """
    الدالة الرئيسية — فقط مرّر مسار الملف
    تكتشف الصيغة تلقائياً وترجع النص الكامل
    """
    fmt = detect_format(file_path)
    loaders = {
        "txt":   load_txt,
        "json":  load_json,
        "jsonl": load_jsonl,
        "csv":   load_csv,
    }
    text = loaders[fmt](file_path)
    print(f"✅ تم تحميل [{fmt.upper()}] | الحجم: {len(text):,} حرف")
    return text

# ── اختبار مباشر ──
if __name__ == "__main__":
    from config import DATA_CONFIG
    text = load_data(DATA_CONFIG["file_path"])
    print(f"أول 200 حرف:\n{text[:200]}")