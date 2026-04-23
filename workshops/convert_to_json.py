import re

VIOLATION_KEYWORD = ["ละเมิด", "ปลอมแปลง", "เลียนแบบ", "ทำซ้ำ", "ดัดแปลง"]


PATENT_KEYWORD = ["สิทธิบัตร", "การประดิษฐ์", "ผังภูมิ"]

COPYRIGHT_KEYWORD = ["ลิขสิทธิ์", "วรรณกรรม", "ดนตรีกรรม", "ศิลปกรรม"]

def detect_category(text):
    is_violation = any(re.search(k,text) for k in VIOLATION_KEYWORD)
    is_patent = any(re.search(k,text) for k in PATENT_KEYWORD)
    is_copyright = any(re.search(k,text) for k in COPYRIGHT_KEYWORD)

    if is_violation:
        if is_patent : return 1
        if is_copyright : return 2
    return 0

sample = "พบการทำซ้ำวรรณกรรม"

print(f"result : {sample}")
print(f"Predicted Class : {detect_category(sample)}")


# Context-aware Confidence


def cal_confidence(text,predicted_class):
    base_conf = 0.70
    signal = []
    if "มาตรา" in text or "พ.ร.บ" in text:
        base_conf += 0.15
        signal.append("statotury_ref")
    if "คำพิพากษา" in text :
        base_conf += 0.10
        signal.append("precednet_ref")
    return min(base_conf,0.99), signal

text_with_context = "ละเมิดลิขสิทธิ์ตามมาตรา 27 แห่ง พ.ร.บ. ลิขสิทธิ์"
conf , sig = cal_confidence(text_with_context,2)
print(f"Confidence: {conf: 2f}")
print(f"Signals: {sig}")
