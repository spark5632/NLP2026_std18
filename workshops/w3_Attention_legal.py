import numpy as np
class SinusodaIPositionEncoding:
    def __init__(self,max_seq_len = 10 , d_model = 16):
        pe = np.zeros((max_seq_len,d_model))
        pos= np.arange(max_seq_len.reshape(-1,1))
        