import numpy as np

MODEL_REGISTRY = {
    "mbert": {
        "hf_name" : "bert-base-multilingual-cased",
        "params"  : "110M",
        "thai_cov": "~1%",
        "note"    : "ใช้ได้หลายภาษา แต่ Thai coverage น้อย",
    },
    "xlmr": {
        "hf_name" : "xlm-roberta-base",
        "params"  : "270M",
        "thai_cov": "~5%",
        "note"    : "RoBERTa architecture, Thai ดีกว่า mBERT",
    },
    "wangchanberta": {
        "hf_name" : "airesearch/wangchanberta-base-att-spm-uncased",
        "params"  : "110M",
        "thai_cov": "100%",
        "note"    : "ดีที่สุดสำหรับ Thai legal text",
    },
}

def show_model_comparison():
    print("=" * 60)
    print("  4.1 เลือก Pretrained Model สำหรับ Thai Legal Text")
    print("=" * 60)
    print(f"  {'Model':<16} {'Params':<8} {'Thai%':<8} หมายเหตุ")
    print(f"  {'─'*56}")
    for name, m in MODEL_REGISTRY.items():
        print(f"  {name:<16} {m['params']:<8} {m['thai_cov']:<8} {m['note']}")
    print(f"\n  → เลือก WangchanBERTa เพราะ pre-train บน Thai Wikipedia + CCNet")

show_model_comparison()

def get_device():
    """ตรวจสอบ GPU อัตโนมัติ: CUDA → MPS → CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print(f"  ✅ Apple Silicon (MPS)")
            return "mps"
        else:
            print(f"  ⚠️  ไม่มี GPU → ใช้ CPU (ช้ากว่า 10-20x)")
            return "cpu"
    except ImportError:
        print("  ℹ️  ไม่มี PyTorch → ใช้ Mock mode")
        return "mock"

DEVICE = get_device()
print(f"\n  Device ที่ใช้: {DEVICE}")


class MockTokenize:
    """จำลอง BERT Tokenizer"""
    LEGAL_TERMS = ["ภูมิปัญญาท้องถิ่น", "สิทธิการประดิษฐ์", "อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ"]

    def encode(self, texts, max_length=24):
        """แปลงข้อความ -> input_ids + attention_mask
           input_ids : [CLS] + token_ids + [SEP] + [PAD]...
           attention_mask: 1 = real token, 0 = padding
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attention_mask = [], []  # ✅ reset ก่อน loop
        for text in texts:
            # [CLS=1] ... [SEP=2]
            ids = [1]+[ord(c)%5000+100 for c in text[:max_length-2]]+[2]
            pad_len = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids  = ids + [0] * pad_len
            input_ids.append(ids)           # ✅ เพิ่ม append ที่หายไป
            attention_mask.append(mask)     # ✅ เพิ่ม append ที่หายไป

        return {
            "input_ids"      : np.array(input_ids,      dtype=np.int32),
            "attention_mask" : np.array(attention_mask, dtype=np.int32),
        }



    def show(self, text, max_length=32): # ← เพิ่ม max_length ให้ยาวพอ 
        enc = self.encode([text], max_length) 
        ids =enc["input_ids"][0] 
        mask = enc["attention_mask"][0] 
        real_len = mask.sum() 
        print(f"\n ข้อความ : '{text}'") 
        print(f" input_ids : {ids.tolist()}") # ← แสดงทั้ง array รวม 0 
        print(f" mask : {mask.tolist()}") 
        print(f" real tokens: {real_len} PAD: {max_length - real_len}") 
        print(f"\n แยกส่วน:") 
        print(f" [CLS] = {ids[0]}") 
        print(f" tokens= {ids[1:real_len-1].tolist()}")
        print(f" [SEP] = {ids[real_len-1]}") 
        print(f" [PAD] = {ids[real_len:].tolist()}") # ← แสดง 0s

# ทดสอบ
tok = MockTokenize()


tok.show("ผู้ต้องหาละเมิดสิทธิบัตร")
print("เพิ่มคำศัพท์เพิ่มเข้าใน Tokenizer")
for term in MockTokenize.LEGAL_TERMS:
    print(f"+{term}")


class MockTokenizer:
    LEGAL_TERMS = ["สิทธิบัตรการประดิษฐ์", "สิทธิบัตร", "ละเมิดสิทธิบัตร", "ผู้ต้องหา"]
    
    def __init__(self):
        # จำลอง Vocab พื้นฐาน
        self.vocab = {"ผู้": 1, "ต้อง": 2, "หา": 3, "ละ": 4, "เมิด": 5, "สิทธิ": 6, "บัตร": 7}

    def show(self, text):
        # จำลองการตัดคำแบบ Subword (ถ้าไม่มีใน Vocab จะแยกส่วน)
        print(f"Input: {text}")
        tokens = []
        if text == "สิทธิบัตรการประดิษฐ์":
            tokens = ["สิทธิ", "##บัตร", "##การ", "##ประ", "##ดิษฐ์"]
        elif text == "ผู้ต้องหาละเมิดสิทธิบัตร":
            tokens = ["ผู้", "ต้อง", "หา", "ละ", "เมิด", "สิทธิ", "บัตร"]
        
        print(f"Tokens: {tokens}")

def vocab_expansion_demo():
    tok = MockTokenizer()

    print("=" * 60)
    print(" STEP 1: ก่อน Expansion — คำเฉพาะทางถูกตัดเป็นชิ้น (Subwords)")
    print("=" * 60)

    tok.show("สิทธิบัตรการประดิษฐ์")

    print("\n" + "=" * 60)
    print(" STEP 2: เพิ่มคำใหม่เข้า Vocab (Vocabulary Expansion)")
    print("=" * 60)
    
    base_vocab_size = 5000
    # สร้าง Vocab ใหม่โดย Map คำกฎหมายกับ ID ต่อจากของเดิม
    new_vocab = {term: base_vocab_size + i for i, term in enumerate(MockTokenizer.LEGAL_TERMS)}
    
    print(f"Vocab เดิม  : {base_vocab_size} คำ")
    print(f"เพิ่มคำใหม่  : {len(new_vocab)} คำ")
    print(f"Vocab รวม   : {base_vocab_size + len(new_vocab)} คำ\n")
    
    for term, idx in list(new_vocab.items())[:3]: # โชว์ตัวอย่าง 3 คำ
        print(f" '{term}' \t→ ID: {idx} (Weight ใหม่ = Random ❗)")

    print("\n" + "=" * 60)
    print(" STEP 3: ทำไมต้อง Warm-up ก่อน Fine-tune?")
    print("=" * 60)
    print("""
    ปัญหา: 
    - คำเดิม (Pretrained)  → Weights มีความหมาย/ทิศทางแล้ว
    - คำใหม่ (Expanded)    → Weights เป็นค่าสุ่ม (Random) ❗

    หาก Fine-tune ทันที: 
    Gradient จากคำใหม่ที่ยัง 'สะเปะสะปะ' จะไปรบกวน Weights เดิมที่ดีอยู่แล้ว 
    ทำให้โมเดลเกิด 'Catastrophic Forgetting' (ลืมความรู้เก่า)

    วิธีแก้ (Warm-up Strategy):
    1. Freeze Backbone: Lock ทุก Layer เปิดแค่ Embedding แล้ว Train เฉพาะคำใหม่
    2. Unfreeze & Low LR: เปิดทุก Layer แต่ใช้ Learning Rate ต่ำๆ เพื่อปรับจูน
    3. Full Fine-tune: Train จน Loss นิ่ง
    """)
    print("=" * 60)
    print(" STEP 4: จำลอง Embedding Weight (Cosine Similarity)")
    print("=" * 60)

    np.random.seed(42)
    d_model = 8 
    # Embedding ของคำเดิม (มีทิศทางที่ชัดเจน)
    old_emb = np.array([0.82, -0.34, 0.56, 0.91, -0.12, 0.67, -0.45, 0.23])
    # คำใหม่ตอนแรก (Random สนิท)
    new_before = np.random.randn(d_model) * 0.02
    # คำใหม่หลัง Warm-up (เริ่มขยับเข้าใกล้กลุ่มคำที่เกี่ยวข้อง)
    new_after = old_emb * 0.6 + np.random.randn(d_model) * 0.1
    
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"คำเดิม 'ละเมิด'  : {old_emb.round(2)}")
    print(f"คำใหม่ (สุ่ม)   : {new_before.round(2)}")
    print(f"คำใหม่ (Warm-up): {new_after.round(2)}")
    print(f"\nCosine Similarity เทียบกับ 'ละเมิด':")
    print(f"- ก่อน Warm-up: {cosine(old_emb, new_before):.4f} (ไม่เกี่ยวข้องกัน)")
    print(f"- หลัง Warm-up: {cosine(old_emb, new_after):.4f} (เริ่มมีความหมายใกล้เคียง)")
    print("=" * 60)


    # cosine similarity
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n cosine similarity กับ 'ละเมิด':")
    print(f" ก่อน warm-up : {cosine(old_emb, new_before):.3f} (ไม่เกี่ยวกัน)")
    print(f" หลัง warm-up : {cosine(old_emb, new_after):.3f} (ใกล้เคียงกัน)")

# --- รันโปรแกรม ---
if __name__ == "__main__":
    vocab_expansion_demo()


#ส่วนที่ 3. BERT Encoder 








# #1. Model Selection & Drvice Detection
# #BERT ถูกเทรน (Pre-Train) MLM 
# #ปรับให้เฉพาะด้าน
# #------------- เลือก model ตามเวลา และขนาดข้อมูล -----------
# import numpy as np

# MODEL_REGISTRY ={
#     "mbert":{
#         "hf_name":"bert-base-multilingual-cased",
#         "params":"110M",
#         "thai_cov":"1%",
#         "note":"ใช้ได้หลายภาษา แต่ thai coverage low"
#     },
#     "xlmr":{
#         "hf_name":"xml-roberta-base",
#         "params":"270M",
#         "thai_cov":"5%",
#         "note":"RoBERTa architecture, thai ดีกว่า mBERT"
#     },
#     "wangchanberta":{
#         "hf_name":"airesearch/wangchanberta-base-att-spm-uncased",
#         "params":"110M",
#         "thai_cov":"100%",
#         "note":"ดีที่สุด thai"
#     }
# }
# def show_model_comperison():
#     print("="*60)
#     print("1. เลือก pretrained model for thai legal text")
#     print("="*60)
#     print(f"{"Model":<16} {"params":<8} {"Thai":<8} หมายเหตุ")
#     print(f"{"-"*56}")
#     for name , m in MODEL_REGISTRY.items():
#         print(f"{name:<16} {m["params"]:<8} {m["thai_cov"]:<8} {m["note"]}")
#     print(f"\n -> เลือก WangchanBERTa เพราะ pre-trin บน thai wikipedia +CCNet")
# show_model_comperison()


# # ── Device Detection (v4-5) ────────────────────────────────────
# def get_device():
#     """ตรวจสอบ GPU อัตโนมัติ: CUDA → MPS → CPU"""
#     try:
#         import torch
#         if torch.cuda.is_available():
#             print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
#             return "cuda"
#         elif torch.backends.mps.is_available():
#             print(f"  ✅ Apple Silicon (MPS)")
#             return "mps"
#         else:
#             print(f"  ⚠️  ไม่มี GPU → ใช้ CPU (ช้ากว่า 10-20x)")
#             return "cpu"
#     except ImportError:
#         print("  ℹ️  ไม่มี PyTorch → ใช้ Mock mode")
#         return "mock"

# DEVICE = get_device()
# print(f"\n  Device ที่ใช้: {DEVICE}")

# #2. Tokenization (BERT ใช้ Subword token แก้ปัญหา oov)

# class MockTokenize:
#     """ จำลอง BERT Tokenizer"""
#     #เพิ่มคำศัพท์ ใน vocab
#     LEGAL_TERMS = [
#         "ภูมิปัญญาท้องถิ่น","สิทธิการประดิษฐ์","อนุสิทธิบัตร","ทรัพย์ทางปัญญา","การละเมิดสิทธิ"
#     ]
#     def encode(self,texts,max_lenght=24):
#         """ แปลงข้อความ -> input_ids+attention_mask
#             input_ids : [CLS] + token_isd + [PAS]...
#             atten_mask: 1 = real token , 0 = padding
#         """
#         if isinstance(texts,str):
#             texts=[texts]


#         input_ids,attention_mask=[],[]
#         for text in texts:
#             ids = [1]+[ord(c)%5000+100 for c in text[:max_lenght-2]]+[2]
#             pad_len=max_lenght-len(ids)
#             mask =[1]*len(ids)+[0]*pad_len
#             ids = ids+[0]*pad_len
#         return{
#             "input_ids": np.array(input_ids,dtype= np.int32),
#             "attention_mask": np.array(attention_mask,dtype=np.int32)
#         }
    
#     def show(self,text,max_len=16):
#         enc=self.encode([text],max_len)
#         ids = enc["input_ids"][0]
#         mask= enc["attention_mask"][0]
#         real_len = mask.sum()
#         print(f"\n ข้อความ: {text}")
#         print(f"input_ids : {ids[:real_len].tolist()} + [PADx{max_len-real_len}]")
#         print(f"mask : {mask[:real_len].tolist()} + [0x{max_len-real_len}]")
#         print(f"[CLS=1] อยู่หน้า,[SEP]=2 อยู่หลัง , PAS = 0 เติมให้ครบ {max_len}")


# tok =MockTokenize()
# # enc= tok.encode(["ผู้ต้องหาละเมิดสิทธิบัตร"],max_lenght=32)
# # print(f"shape: {enc["input_ids"].shape}")
# # tok.show("จำเลยละเมิดสิทธิบัตร")

# tok.show("ผู้ต้องหาละเมิดสิทธิบัตร")

# # print()