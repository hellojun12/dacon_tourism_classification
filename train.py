
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CustomDataset
from transform import TrainTransform, TestTransform
from model import CustomModel, BERTClassifier
from trainer import trainer
from utils import load_config, seed_everything

import os

#Set Config
CFG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

#Data Load & Split
data_root = '/home/junshick/Workspace/Dataset/Competition_datasets/Dacon_dataset'
all_df = pd.read_csv(os.path.join(data_root, './train.csv'))
train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.2, random_state=CFG['seed'])

#Label-Encoding
le = preprocessing.LabelEncoder()
le.fit(all_df['cat3'].values)
num_classes = len(le.classes_)

train_label = le.transform(train_df['cat3'].values)
val_label = le.transform(val_df['cat3'].values)
print(f"Total number of class is..... {num_classes}")

#Vectorizer
# vectorizer = CountVectorizer(max_features=4096)
# train_vectors = vectorizer.fit_transform(train_df['overview'])
# train_vectors = train_vectors.todense()

# val_vectors = vectorizer.transform(val_df['overview'])
# val_vectors = val_vectors.todense()

#Tokenizer
print('Initiallizing KoBERT Tokenizer')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
print(train_df.dtypes)
train_inputs = tokenizer.batch_encode_plus(train_df['overview'], max_length=128, padding=True, truncation=True)
val_inputs = tokenizer.batch_encode_plus(val_df['overview'], max_length=128, padding=True, truncation=True)

#Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Set seed
seed_everything(CFG['seed'])

#Set Augmentation
train_transform = TrainTransform(CFG['image_size'], CFG['image_size'])
test_transform = TestTransform(CFG['image_size'], CFG['image_size'])

#Set Data
train_dataset = CustomDataset(data_root, train_df['img_path'].values, train_inputs, train_label, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=1, drop_last=True)

val_dataset = CustomDataset(data_root, val_df['img_path'].values, val_inputs, val_label, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=1, drop_last=True)

#Set Model
#model = CustomModel(num_classes)
base_model = BertModel.from_pretrained('skt/kobert-base-v1')
model = BERTClassifier(base_model, num_classes=num_classes, dr_rate=0.5)
model.eval()

optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["learning_rate"])
criterion = nn.CrossEntropyLoss().to(device)
scheduler = None

trainer(model, CFG['epochs'], optimizer, criterion, train_loader, val_loader, scheduler, device)