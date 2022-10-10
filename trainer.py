import tqdm
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

def validation(model, criterion, val_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = [] 
    
    val_loss = []
    
    with torch.no_grad():
        for img, text, label in tqdm(iter(val_loader)): 
            img = img.float().to(device)
            text = text.to(device)
            label = torch.Tensor(label, dtype=torch.LongTensor)
            label = label.to(device)
            
            model_pred = model(img, text)
            
            loss = criterion(model_pred, label) 
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    test_weighted_f1 = f1_score(true_labels, model_preds, average='weighted') # 실제 라벨값들과 예측한 라벨값들에 대해 f1 점수 계산
    test_acc = accuracy_score(true_labels, model_preds)
    return np.mean(val_loss), test_acc, test_weighted_f1 

def trainer(model, epochs, optimizer, criterion, train_loader, val_loader, scheduler, device):
    model.to(device) # gpu(cpu)에 적용

    best_score = 0
    best_model = None 
    
    for epoch in range(1, epochs+1):
        model.train() 
        t_loss = []
        t_pred = []
        t_label = []
        
        for idx, (img, input_id, attention_mask, label) in enumerate(train_loader):
            #img = img.float().to(device)

            input_id = input_id.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            
            optimizer.zero_grad() 

            model_pred = model(input_id, attention_mask)
            
            loss = criterion(model_pred, label) 
            loss.backward() 
            optimizer.step() 

            t_loss.append(loss.item())
            t_pred.extend(model_pred.argmax(1).detach().cpu().numpy())
            t_label.extend(label.detach().cpu().numpy())
            #logging
            if (idx + 1) % 50 == 0:

                train_loss = np.mean(t_loss)
                train_acc = accuracy_score(t_label, t_pred)
                train_f1 = f1_score(t_label, t_pred, average='weighted')

                 # print train loss
                print(f"Epoch: {epoch:0{len(str(epochs))}d}/{str(epochs)} ", 
                        f"[{idx + 1:0{len(str(len(train_loader)))}d}/{len(train_loader)}]")
                print(f"training loss: {train_loss:>4.4f} training acc: {train_acc:>4.4f} training f1: {train_f1:>4.4f}\n")
                
                t_loss = []
                t_pred = []
                t_label = []
            
        # val_loss, val_acc, val_score = validation(model, criterion, val_loader, device)
            
        # print(f'Epoch [{epoch}], validation loss : [{val_loss:.5f}] validation acc: [{val_acc:.5f}] validation f1 : [{val_score:.5f}]')
        
        # if scheduler is not None:
        #     scheduler.step()
            
        # if best_score < val_score: 
        #     best_score = val_score
        #     best_model = model
        #     print(f'New best model found!! : Epoch [{epoch}], current best f1 score [{val_score:.5f}] ')


