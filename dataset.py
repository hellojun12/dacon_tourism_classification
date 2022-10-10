import torch
from torch.utils.data import Dataset
import cv2
import os

# Dataset 생성
class CustomDataset(Dataset):
    def __init__(self, data_root, encoded_tokens, label_list, infer=False):
        self.data_root = data_root
        self.input_ids = encoded_tokens['input_ids']
        self.attention_mask = encoded_tokens['attention_mask']
        self.label_list = label_list
        self.infer = infer
        
    def __getitem__(self, index):
        # NLP
        input_id = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        
        # Label
        if self.infer: 
           
            return input_id, attention_mask
        else: # inf

            label = self.label_list[index]
            return torch.LongTensor(input_id), torch.LongTensor(attention_mask), label
        
    def __len__(self):
        return len(self.img_path_list)