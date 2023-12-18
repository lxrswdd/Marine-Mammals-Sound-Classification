import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn import preprocessing
import mytools
from FSMN import CVFSMNv2
import torch.nn.functional as F
import x_vector
from sklearn.preprocessing import StandardScaler
import shutil
from sklearn.metrics import precision_recall_fscore_support
from timm.loss import LabelSmoothingCrossEntropy
import sys
from XCA import XCA
import random


import time 

verbose = False
batch_size = 32
learning_rate= 0.00001
nhid=128
device = "cuda:1"

for learning_rate in [0.00001]:

    for dataset in [ 'moby']:

        log_dir = f'./logs/xfsmn/{dataset}'
        os.makedirs(log_dir, exist_ok=True)
        log = open(f"{log_dir}/xfsmn_lr{learning_rate}_Aug13.log", "w")
        sys.stdout = log

        if dataset == 'combined':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/combined/8khz/train/combined_data.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/combined/8khz/test/combined_data.npz')

        elif dataset == 'moby':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/Moby/8khz/train/combined_data.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/Moby/8khz/test/combined_data.npz')

        elif dataset == 'watkins':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/WATKINS/8khz/train/combined_data.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/WATKINS/8khz/test/combined_data.npz')
        elif dataset =='watkins12':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Waktins12/features/train/combined_data.npz'
            testset  = np.load('/data2/xiangrui_d2/MMC/Dataset/Waktins12/features/test/combined_data.npz')
        checkpoint_dir = f'./checkpoints/xfsmn_checkpoints/8khz/{dataset}/Aug_13/teacher_model3/lr{learning_rate}'
        os.makedirs(checkpoint_dir, exist_ok=True)


        class teacher_model(nn.Module):
                def __init__(self,ninp,class_num,nhid=256) :
                    super().__init__()
                    x_vector_dim = 256
                    self.cvfsmn2 = CVFSMNv2(memory_size = 256,input_size = ninp,output_size = nhid, projection_size = 128)
                    self.cvfsmn3 = CVFSMNv2(memory_size = 128,input_size = nhid,output_size = nhid, projection_size = 64)
                    self.xca  = XCA(nhid,num_heads=8,attn_drop=0.2,proj_drop=0.2) # The output shape of xca is same as the input
                    self.layernorm = nn.LayerNorm(nhid)
                    self.x_vector = x_vector.X_vector(input_dim=seg,num_classes=class_num,output_dim=x_vector_dim) # output of Xvector is 512 dim.
                    self.insnorm = nn.InstanceNorm1d(256)
                    # self.densenet_embedding = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
                    self.densenet_embedding = torchvision.models.densenet121(weights = "DenseNet121_Weights.DEFAULT")
                    self.densenet_embedding.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    self.densenet_embedding.classifier = nn.Identity()
                    # for param in self.densenet_embedding.parameters(): #freeze model
                    #     param.requires_grad = False

                    dnns = []
                    dnns.extend([nn.Linear(nhid*seg+x_vector_dim+1024,2048), nn.GELU(),torch.nn.Dropout(0.3)])
                    dnns.extend([nn.Linear(2048,1024), nn.GELU(),torch.nn.Dropout(0.3)])
                    dnns.extend([nn.Linear(1024,class_num)])           
                    self.dnn_layers = nn.Sequential(*dnns)            
                    
                def forward(self,X):
                    X_CNN_input = torch.unsqueeze(X,1) 
                    cnn_out = self.densenet_embedding(X_CNN_input)
                    xvector = self.x_vector(X)
                    out  = self.cvfsmn2(X)
                    out = self.insnorm(out)
                    out = self.layernorm(self.xca(out))
                    ccfsmn_out = out.reshape((-1,out.shape[1]*out.shape[2]))
                    out = torch.hstack((ccfsmn_out,xvector,cnn_out))
                    out = self.dnn_layers(out)
                    
                    return out
                
        class teacher_model2(nn.Module):
            def __init__(self,ninp,class_num,nhid=256) :
                super().__init__()
                x_vector_dim = 256
                self.cvfsmn = CVFSMNv2(memory_size = 512,input_size = ninp,output_size = nhid, projection_size = 256)
                self.cvfsmn2 = CVFSMNv2(memory_size = 256,input_size = ninp,output_size = nhid, projection_size = 128)
                self.cvfsmn3 = CVFSMNv2(memory_size = 128,input_size = nhid,output_size = nhid, projection_size = 64)
                self.xca  = XCA(nhid,num_heads=8,attn_drop=0.2,proj_drop=0.2) # The output shape of xca is same as the input
                self.layernorm = nn.LayerNorm(nhid)
                self.x_vector = x_vector.X_vector(input_dim=seg,num_classes=class_num,output_dim=x_vector_dim) # output of Xvector is 512 dim.
                self.insnorm = nn.InstanceNorm1d(256)
                # self.densenet_embedding = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
                self.densenet_embedding = torchvision.models.densenet121(weights = "DenseNet121_Weights.DEFAULT")
                self.densenet_embedding.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.densenet_embedding.classifier = nn.Identity()
                # for param in self.densenet_embedding.parameters(): #freeze model
                #     param.requires_grad = False
                dnns = []
                dnns.extend([nn.Linear(nhid*seg+x_vector_dim+1024,2048), nn.GELU(),torch.nn.Dropout(0.3)])
                dnns.extend([nn.Linear(2048,128), nn.GELU(),torch.nn.Dropout(0.3)])
                dnns.extend([nn.Linear(128,class_num)])           
                self.dnn_layers = nn.Sequential(*dnns)            
                
            def forward(self,X):
                X_CNN_input = torch.unsqueeze(X,1) 
                cnn_out = self.densenet_embedding(X_CNN_input)
                xvector = self.insnorm(self.x_vector(X))
                out  = self.cvfsmn(X)
                out = self.xca(out)
                out = self.xca(out)
                ccfsmn_out = out.reshape((-1,out.shape[1]*out.shape[2]))
                out = torch.hstack((ccfsmn_out,xvector,cnn_out))
                out = self.dnn_layers(out)
                
                return out
            
        class teacher_model3(nn.Module):
            def __init__(self,ninp,class_num,nhid=256) :
                super().__init__()
                x_vector_dim = 256
                # self.cvfsmn = CVFSMNv2(memory_size = 512,input_size = ninp,output_size = nhid, projection_size = 256) # output (batch,length,nhid)
                self.cvfsmn2 = CVFSMNv2(memory_size = 256,input_size = ninp,output_size = nhid, projection_size = 128)
                self.cvfsmn3 = CVFSMNv2(memory_size = 128,input_size = nhid,output_size = nhid, projection_size = 64)
                
                self.xca  = XCA(nhid,num_heads=8,attn_drop=0.1,proj_drop=0.1) # The output shape of xca is same as the input
                self.layernorm = nn.LayerNorm(nhid)
                self.x_vector = x_vector.X_vector(input_dim=seg,num_classes=class_num,output_dim=x_vector_dim) # output of Xvector is 512 dim.
                self.insnorm = nn.InstanceNorm1d(256)

                self.densenet_embedding = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
                self.densenet_embedding.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.densenet_embedding.classifier = nn.Identity()
                for param in self.densenet_embedding.parameters(): #freeze model
                    param.requires_grad = False
                dnns = []
                dnns.extend([nn.Linear(nhid*seg+x_vector_dim+1024,2048), nn.GELU(),torch.nn.Dropout(0.2)])
                dnns.extend([nn.Linear(2048,1024), nn.GELU(),torch.nn.Dropout(0.2)])
                dnns.extend([nn.Linear(1024,class_num)])           
                self.dnn_layers = nn.Sequential(*dnns)       
                    
            def forward(self,X):
                X_CNN_input = torch.unsqueeze(X,1) 
                cnn_out = self.densenet_embedding(X_CNN_input)
                # print(cnn_out.shape)
                xvector = self.x_vector(X)

                out  = self.cvfsmn2(X)
                out = self.cvfsmn3(out)
                out = self.xca(out)
                out = out.reshape((-1,out.shape[1]*out.shape[2]))
                ccfsmn_out = self.insnorm(out)

                out = torch.hstack((ccfsmn_out,xvector,cnn_out))
                out = self.dnn_layers(out)
                
                return out

        # for dataset in ['moby', 'watkins', 'combined']:


        print('----------------------------------')
        print('seperation-seperation-seperation')
        print('----------------------------------')
        print('dataset: ', dataset)

        f1s = []
        for _ in range(5):
            print('-----------------------------------------------------------------')
            random_state_manual = random.randint(0,1000)
            print('random state: ',random_state_manual)

            curr_dir = os.getcwd()
            data = np.load(root_dir)
            X = data['arr_0']
            y = data['arr_1']

            le = preprocessing.LabelEncoder()
            # class_list = list(counter1.keys())
            if dataset == 'watkins12':
                class_list =  ['sperm whale','Finback Whale','Humpback Whale','Killer Whale','Short-Finned (Pacific) Pilot Whale','Long-Finned Pilot Whale','Pantropical Spotted Dolphin','Spinner Dolphin','Common Dolphin','Bottlenose Dolphin','Weddell Seal','Bowhead Whale']
            else:
                class_list = ['Blue Whale','Finback Whale','Minke Whale','Humpback Whale','Bowhead Whale']
            le.fit(class_list)
            y = le.transform(y)
            list(le.classes_)

            # Scaler fit the X_train
            seg = X.shape[1]
            fdim = X.shape[2]
            class_num = len(class_list)
            X = X.reshape((-1,seg*fdim))
            scaler = StandardScaler()
            scaler.fit_transform(X)
            X = X.reshape(-1,seg,fdim)

            
            X_test2 = testset['arr_0']
            y_test2 = testset['arr_1']

            y_test22 = le.transform(y_test2)
            
            # Scaler fit the X_test
            X_test2 = X_test2.reshape((-1,seg*fdim))
            scaler.transform(X_test2)
            X_test2 = X_test2.reshape(-1,seg,fdim)
            
            test_dataset2 = mytools.mydata(X_test2,y_test22)
            testloader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, num_workers=0)

        
            def train_model(model,epoch,trainloader,validloader,optimizer,loss_function,X_train,X_valid,ES_patience=5):
                
                valid_loss_list = []
                train_loss_list = []
                valid_loss_plot_list = []
                train_loss_plot_list = []
                valid_acc_list = []
                train_acc_list = []
                y_true_list = []
                y_pred_list = []
                F1_list = []

                earlystopper = mytools.EarlyStopping(patience=ES_patience,verbose=False,path=os.path.join(os.getcwd(),'xfsmn_checkpoint.pt'))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1,  verbose=False)

                for epoch in range(epoch):
                    # print('epoch: ',epoch)
                    train_loss = 0.0
                    model.train()
                    train_acc = 0
                    valid_acc = 0
                    train_correct = 0
                    valid_correct = 0
                    for i, data in enumerate(trainloader):

                        X, y = data
                        y = y.long()

                        if torch.cuda.is_available():
                            # print('running on cuda')
                            X, y = X.to(device), y.to(device)
                            model.to(device)
                        else:
                            print('running on cpus')
                        optimizer.zero_grad()
                        y_pred= model(X)   
                        # calculate loss function
                        loss = loss_function(y_pred, y)    
                        loss.backward()

                        #Calculate Train Accuracy
                        y_pred_cpu = y_pred.cpu().detach()
                        y_detached = y.cpu().detach()
                        train_correct += (y_pred_cpu.argmax(axis=1) == y_detached).float().sum()

                        # update with current step regression parameters 
                        optimizer.step()
                        # if (i+1) % 10 == 0:
                        #     scheduler.step()
                        # scheduler.step()
                        train_loss += loss.item()
                    train_loss_list.append(train_loss/ len(trainloader))
                    
                    valid_loss = 0.0

                    
                    model.eval()   
            
                    for X, y in validloader:
                        y = y.long()
                        if torch.cuda.is_available():
                            X, y = X.to(device), y.to(device)

                            
                        y_pred = model(X)
                        
                        # y_pred = y_pred.long()
                        loss = loss_function(y_pred,y)
                        valid_loss = loss.item() * X.size(0)
                    
                        # Caculate Validation Accuracy
                        y_pred_cpu = y_pred.cpu().detach()
                        y_detached = y.cpu().detach()
                        valid_correct += (y_pred_cpu.argmax(axis=1) == y_detached).float().sum()

                    epoch_valid_loss = valid_loss/ len(validloader)
                    earlystopper(epoch_valid_loss,model,epoch)
                    if earlystopper.early_stop:
                        print("Early stopping")
                        break

                    valid_loss_list.append(epoch_valid_loss)

                    train_acc = 100 * (train_correct / X_train.shape[0])
                    valid_acc = 100 * (valid_correct / X_valid.shape[0])
                    valid_acc_list.append(valid_acc)
                    train_acc_list.append(train_acc)
                    # F1_list.append(precision_recall_fscore_support(torch.tensor(y_pred_cpu), torch.tensor(y_detached).int(), average='macro'))
                    if verbose:
                        if epoch %1 ==0:
                            print(f'Epoch {epoch+1} \t\t Training Accuracy: {train_acc} \t\t Validation Accuracy: {valid_acc}')
                            print(f' \t\t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}\n')
                            print('---------------------------------------------------------------------')
                # plt.plot(valid_loss_list)
                # plt.plot(train_loss_list)
                # plt.show()
                return train_acc_list,valid_acc_list,train_loss_list,valid_loss_list

            cross_validations = []
            cross_validation_train = []
            F1_Score = []
            print('Running random state:',random_state_manual)
            X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=random_state_manual)
                

            # print(X_train.shape)
            train_dataset = mytools.mydata(X_train,y_train)
            valid_dataset = mytools.mydata(X_valid,y_valid)
            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            
            model = teacher_model3(ninp=X.shape[2],class_num=class_num,nhid=nhid)
            
            total_params =  sum(p.numel() for p in model.parameters())
            print(f"{total_params:,} total parameters.")

            loss_function = LabelSmoothingCrossEntropy()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_time = time.time()
            model.train()
            train_acc_list,valid_acc_list,train_loss_list,valid_loss_list = train_model(model,300,trainloader,validloader,optimizer,loss_function,X_train,X_valid,ES_patience=5)
            print('Training time: ',time.time()-train_time)

            y_pred = []
            y_true = []
            compare = []
            # iterate over test data
            
            model.load_state_dict(torch.load("./xfsmn_checkpoint.pt"))
            model.eval()
            inference_time = time.time()
            for X, y in testloader2:
                X = X.to(device)
                output = model(X) # Feed Network
                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction
                y = y.data.cpu().numpy()
                y_true.extend(y) # Save Truth

            for a,b in zip(y_pred,y_true):
                compare.append(a==b)
            print('testset accuracy: ',sum(compare)/len(y_true)*100)
            print('F1s',precision_recall_fscore_support(torch.tensor(y_pred), torch.tensor(y_true).int(), average='macro'))
            f1 = precision_recall_fscore_support(torch.tensor(y_pred), torch.tensor(y_true).int(), average='macro')
            
            # if f1[-2] >0.85:
            f1s.append(f1[:3])
            
            print('Inference time: ',time.time()-inference_time)

            shutil.copy('./xfsmn_checkpoint.pt',f'{checkpoint_dir}/checkpoint_{f1[-2]}.pt')

        f1_ave = np.array(f1s)
        f1_ave = np.mean(f1_ave,axis=0)
        print(f'The average f1 for dataset {dataset} is:',f1_ave)
