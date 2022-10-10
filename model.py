import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Image
        self.cnn_extract = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                # input_channel = 3 : RGB 3개의 채널이기 때문
                # out_channel = 8 : 출력하는 채널 8개
                # stride = 1 : stride 만큼 이미지 이동하면서 cnn 수행
                # padding = 1 : zero-padding할 사이즈
            nn.ReLU(), # 사용할 활성화 함수: Relu를 사용
            nn.MaxPool2d(kernel_size=2, stride=2), # 최댓값을 뽑아내는 맥스 풀링
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Text
        self.nlp_extract = nn.Sequential(
            nn.Linear(4096, 2048), # 선형회귀. 4096개의 입력으로 2048개의 출력
            nn.ReLU(),
            nn.Linear(2048, 1024), # 선형회귀. 2048개의 입력으로 1024개의 출력
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
            # 선형회귀. 4160개의 입력으로 num_classes, 즉 cat3의 종류 개수만큼의 출력
            # 근데 왜 4160개? "4160 - 1024 = 3136"이고 "3136 / 64 = 49". 즉 이미지는 "7*7*64"로 출력됨.
        )
            

    def forward(self, img, text):
        img_feature = self.cnn_extract(img) # cnn_extract 적용
        img_feature = torch.flatten(img_feature, start_dim=1) # 1차원으로 변환
        text_feature = self.nlp_extract(text) # nlp_extract 적용
        feature = torch.cat([img_feature, text_feature], axis=1) # 2개 연결(3136 + 1024)
        output = self.classifier(feature) # classifier 적용
        return output

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=127,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    # def gen_attention_mask(self, token_ids, valid_length):
    #     attention_mask = torch.zeros_like(token_ids)
    #     for i, v in enumerate(valid_length):
    #         attention_mask[i][:v] = 1
    #     return attention_mask.float()

    def forward(self, token_ids, attention_mask):

        output =  self.bert(input_ids = token_ids, attention_mask = attention_mask)
        pooler = output.pooler_output
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)