
# 1. สร้างพจณานุกรม

import random
import numpy as np
from imblearn.over_sampling import SMOTE ,  RandomOverSampler
from collections import Counter 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report


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
        sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)
    X_res , y_res = sampler.fit_resample(X,y)
    print(f"Balanced distribution: {Counter(y_res)}")
    return X_res , y_res

# จำลองข้อมูล imbalance 
X_mock = np.random.randn(12,5)
y_mock = np.array([0]*8 + [1]*2 + [2]*2)

X_res , y_res = balance_legal_data(X_mock,y_mock)


#เปลี่ยนเป็น3 miti

X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1)
y_res_tensor = torch.tensor(y_res, dtype=torch.long)


print(f" shape 2d : {X_res.shape}")
print(f" shape 3d : {X_res_3d.shape}")


#----- 3. BiLSTM  ------------------------------------------------------------


class LegalBiLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3):
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
sample_input = torch.randn(1,5,5) #  1 doc 5 
output = model(sample_input)
print(f"--BILSTM Output")
print(f"Logics: {output.detach().numpy()}")

#ปรับมิติ
# X_train = X_res.unsqueeze(1)

#ตรวจ




# 3.2 LSTM
class LegalLSTM(nn.Module):
    def __init__(self,input_dim=5,hidden_dim=32,output_dim=3):
        super(LegalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=False)
        self.fc = nn.Linear(hidden_dim,output_dim)
        nn.init.xavier_uniform(self.fc.weight)#เลือกค่าเหมาะสม สำหรบเทรนรอบแรก

    def forward(self,x):
        if x.dim()==2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out,dim=1)
        return self.fc(pooled)
        
#3.3 เทรนโมเดล LSTM VS LegalBiLSTM
def train_and_evaluate(model_class,name, X,y,Class_names):
    if X.dim()==2:
        X = X.unsqueeze(1)
    # print(f"Training  : {name}...")
    model = model_class()
    # Cost-Sensitive Weight (FN = Fals Negative)
    #0 ไม่ผิด 1 ละเมิดสิทธิบัตร 2 ละเมิดลิขสิทิ์
    weights = torch.tensor([1.0,2.0,2.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    #sample training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    #ประเมิน evaluate with cm
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(model(X),dim=1).numpy()
    cm = confusion_matrix(y,y_pred)

    #heat map
    # plt.figure(figsize=(5,4))
    fig,ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues" if "Bi" in name else "Purples"
                ,xticklabels=Class_names,yticklabels=Class_names,ax=ax )
    ax.set_title(f"Confusion Matrix : {name}")
    ax.set_ylabel(f"Actual")
    ax.set_xlabel(f"Predicted")
    plt.tight_layout()
    plt.show()
    return classification_report(y,y_pred,target_names=Class_names)


# รันเทียบ training & evaluate
class_list = ["NO-INF","PATEIT","COPYRIGHT"]

#X-train,Y-train = ไม่ผ่าน mote , X_res ,y_res = ผ่านการ SMOTE แล้ว
report_lstm = train_and_evaluate(LegalLSTM,"Unidirextional LSTM", X_res_3d,y_res_tensor,class_list)
report_bilstm = train_and_evaluate(LegalBiLSTM,"Unidirextional LSTM", X_res_3d,y_res_tensor,class_list)
#ตรวจ 
print(f"X_res Shape: {X_res.shape}")





# Confusion matrix Visualization