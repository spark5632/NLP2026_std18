import re
# import deepcut

LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด"]
def legal_tokenizer(text):
    # ใช้ Regex จัดการเบื้องต้น ค่อยตัดส่วนที่เหลือ
    # pattern = "|".join(map(re.escape,LEGAL_KEYWORD))
    # tokens = re.findall(pattern + r"|\ไ+",text)
    # return tokens
    compound = "|".join(map(re.escape,sorted(LEGAL_KEYWORD,key = len,reverse=True),))
    pattern = (compound + r"|\u0E00-\u0E7F" + r"|[a-zA-Z0-9]+")
    return re.findall(pattern,text)
    

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)

print(f"input: {test_text}")
print(f"output: {tokens}")

# print(f"OUtput deep:{deepcut.tokenize(test_text)}")