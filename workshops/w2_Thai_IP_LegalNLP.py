# 1. สร้างพจณานุกรม

import random
import numpy as np
from imblearn.over_sampling import SMOTE ,  RandomOverSampler
from collections import Counter 

LEGAL_SYNONYMS = {
    "ละเมิด":["ฝ่าฝืน","กระทำผิด","ล่วงสิทธิ"],
    "จำหน่าย":["ขาย","เผยแพร่","กระจายสินค้า"],
    "ปลอมแปลง":["ทำเทียม","เลียนแบบ"]
}


def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    for i , word in enumerate(words):
        if word in  LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return "".join(new_words)
original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original)
print(f"--Data Augmentatation---")
print(f"Original : {original}")
print(f"Augment : {augmented}")

#--------------------------------------------------------------------------------------------------

def balance_legal_data(X,y):
    counts = Counter(y)
    print(f"Original distribution : {counts}")
    #ตัวน้อย
    min_samples = min(counts.values())
    if min_samples > 1 :
        sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=18)
    else:
        sampler = RandomOverSampler(random_state=18)
    X_res , y_res = sampler.fit_resample(X,y)
    print(f"Balanced distribution: {Counter(y_res)}")
    return X_res , y_res

# จำลองข้อมูล imbalance 
X_mock = np.random.randn(12,5)
y_mock = np.array([0]*10+[1]*2)

X_res , y_res = balance_legal_data(X_mock,y_mock)

#----- 3. BiLSTM  ------------------------------------------------------------

import torch 
import torch.nn as nn
class LegalBiLSTM(nn.Module):
    def __init__(self,input_dim=16, hidden_dim=32,output_dim=3):
        super(LegalBiLSTM,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        # Mean Pooling 
        pooled = torch.mean(lstm_out,dim=1)
        return self.fc(pooled)

model = LegalBiLSTM()
sample_input = torch.randn(1,5,16) #  1 doc 5 
output = model(sample_input)
print(f"--BILSTM Output")
print(f"Logics: {output.detach().numpy()}")