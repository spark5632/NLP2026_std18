import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from transformers import AutoTokenizer
# from pythainlp.tokenize.attacut import AttacutTokenizer  as attacut
# import deepcut

LEGAL_KEYWORDS = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด"]

def legal_tokenizer(text):
    # 1.Protect Compound Keywords ด้วย Placeholder
    sorted_kw = sorted(LEGAL_KEYWORDS,key=len,reverse=True)
    placeholders = {}
    protected = text
    for i, kw in enumerate(sorted_kw):
        ph = f"__KW{i}__"
        if kw in protected:
            placeholders[ph] = kw
            protected = protected.replace(kw,ph)
    # 2. tokenize ด้วย pythainlp
    tokens_raw = word_tokenize(protected,engine="newmm",keep_whitespace=False)
    
    # 3. restore placeholder
    return [placeholders.get(t,t) for t in tokens_raw]

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"Input: {test_text}")
print(f"Output: {tokens}")


#การวัดค่า ความกำกวม
def calculate_baseline_ambiguity(text):
    matches = []
    for word in LEGAL_KEYWORDS:
        for m in re.finditer(re.escape(word),text):
            if m:
                matches.append((m.start(),m.end(),word))
            #ตรวจสอบการทับซ้อน
    overlaps = 0
    for i in range(len(matches)):
        for j in range   (i+1,  len(matches)):
            if matches[i][0] < matches[j][1] and matches [j][0] < matches[i][1]:
                overlaps += 1
    return overlaps/len(matches) if matches else 0
    
sample_text = "คดีการละเมิดสิทธิบัตร"
baseline_tokens = legal_tokenizer(sample_text)
baseline_rate = calculate_baseline_ambiguity(sample_text)
print(f"W1 Baseline Rusrult")
print(f"Tokens : {baseline_tokens} ")
print(f"Baseline Ambiguity : {baseline_rate}")

# WangChanBERTa Pretain
#  1.Load WangchanBerta


model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def berta_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return [t.replace("","")for t in tokens if t.replace("","")]

# test กำกวม 
def analyze_refined_amiguity(text,legal_keywords):
    tokens = berta_tokenizer(text)
    frag_score = []
    for kw in legal_keywords:
        if kw in text:
            kw_tokens = berta_tokenizer(kw)
            fragment_ratio = len(kw_tokens)/1
            frag_score.append(fragment_ratio)
    # Ambiguity = Average Fragmentation -1 แต่ถ้าตัดพอดี เท่ากับ 0
    avg_frag = (sum(frag_score) / len(frag_score)) -1 if frag_score else 0
    return min(avg_frag,1.0)
    # run แสดงผล
refined_tokens = berta_tokenizer(sample_text)
refined_rate = analyze_refined_amiguity(sample_text,LEGAL_KEYWORDS)
print(f"--W1 : Refined With WangchanBERTa---")
print(f"Tokens : {refined_tokens}")
print(f"New Ambiguity Fragmentation Rate : {refined_rate:.4f}")



#2.context-aware entity extraction
def extract_legal_entities(text):
    entities = []
    # จำลองหาความผิด ประเภทของ IP_TYPE และหาการกระทำ action 
    if "สิทธิบัตร" in text:
        entities.append({"type":"IP_TYPE","value":"PATENT","conf":0.95})
    if "ละเมิด" in text:
        entities.append({"type":"ACTION","value":"INFRINGEMENT  ","conf":0.85})
    return entities 
      

sample = "มีการละเมิดสิทธิบัตรเกิดขึ้นในเขตพื้นที่"
found  = extract_legal_entities(sample)
print(f"Entity Extraction")
for e in found:
    print(f"{e["type"]}{e["value"]}(confident{e["conf"]})")

#3 feature engineering (TF-IDF Base)
corpus = [
    "ละเมิดสิทธิบัตร เครื่องหมายการค้า",
    "การกระทำความผิด ลิขสิทธิ์",
    "จำเลย ละเมิด ลิขสิทธิ์"
]
# สร้าง Vectorizer โดยใช้ Tokenizer ที่สร้างเอง
vectorizer = TfidfVectorizer(tokenizer=legal_tokenizer,token_pattern=None)
tfudf_matrix = vectorizer.fit_transform(corpus)

print (f"---TF-IDF--Vector (Shape:{tfudf_matrix.shape})-------")
print(f"Vocabulary:{vectorizer.get_feature_names_out()}")
print(f"Vector sample (Doc 1):\n{tfudf_matrix[1].toarray()}")


# 4.Physics Gate Weight (Legal Hierachy)
def compute_physics_gate_weight(entities):
    base_weight = 5.0
    for e in entities:
        if e["value"] == "PATENT" : base_weight += 2.0
        if e["value"] == "INFRINGEMENT": base_weight += 1.5
    return min(base_weight,10.0)#max

weight = compute_physics_gate_weight(found)
print(f"--Physics Gate Bridge --")
print(f"Legal Context Weight : {weight:.2f}/10")
print(f"Statys: {"Hight Alert - Trigger Sensor" if weight >= 7 else "Normal Monitoring" }")



#---------------------------------------------------------------------------------------------
# LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด"]
# def legal_tokenizer(text):
#     compound = "|".join(map(re.escape,sorted(LEGAL_KEYWORD,key = len,reverse=True),))
#     pattern = (compound + r"|\u0E00-\u0E7F" + r"|[a-zA-Z0-9]+")
#     return re.findall(pattern,text)
#     ## ใช้ Regex จัดการเบื้องต้น ค่อยตัดส่วนที่เหลือ
#     ## pattern = "|".join(map(re.escape,LEGAL_KEYWORD))
#     ## tokens = re.findall(pattern + r"|\ไ+",text)
#     ## return tokens
    

# test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
# tokens = legal_tokenizer(test_text)

# print(f"input: {test_text}")
# print(f"output: {tokens}")

# ## print(f"OUtput deep:{deepcut.tokenize(test_text)}")
#---------------------------------------------------------------------------------------------




