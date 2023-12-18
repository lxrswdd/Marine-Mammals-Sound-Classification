import os
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
import torch.nn.utils.prune
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import mytools
from YRQ_fsmn import CVFSMNv2
import torch.nn.functional as F
from collections import Counter
import x_vector
from sklearn.preprocessing import StandardScaler
import shutil
from sklearn.metrics import precision_recall_fscore_support
from timm.loss import LabelSmoothingCrossEntropy
import sys
from XCA import XCA
import nets
import model_soup
from mytools import EarlyStopping
import time 
import random

verbose = False
batch_size = 32
nhid=128
epoch=300
F1_score = 0
device = 'cuda:1'

class teacher_model(nn.Module):
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
for learning_rate in [0.00002]:
    for dataset in [ 'watkins12']:

        if dataset == 'combined':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/combined/8khz/train/combined_data.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/combined/8khz/test/combined_data.npz')

        elif dataset == 'moby':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/Moby/8khz/train/combined_data.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/Moby/8khz/test/combined_data.npz')

        elif dataset == 'watkins':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Combine/features/WATKINS/8khz/train/combined_dat a.npz'
            testset = np.load('/data2/xiangrui_d2/MMC/Dataset/Combine/features/WATKINS/8khz/test/combined_data.npz')

        elif dataset =='watkins12':
            root_dir = '/data2/xiangrui_d2/MMC/Dataset/Waktins12/features/train/combined_data.npz'
            testset  = np.load('/data2/xiangrui_d2/MMC/Dataset/Waktins12/features/test/combined_data.npz')

        log = open(f"/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/logs/KD/Cooling/{dataset}/KD_cooling_{dataset}_lr{learning_rate}_Aug14_128d.log", "w")
        sys.stdout = log
        checkpoint_dir = f'/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/kd_checkpoints/8khz/{dataset}/temperature_test/cooling/Aug14_128d_lr{learning_rate}'
        print('save checkpoint at: ', checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        print('----------------------------------')
        print('seperation-seperation-seperation')
        print('----------------------------------')
        print('dataset: ', dataset)

        f1s = []
        while len(f1s) < 5:
            train_time_list = []
            infer_time_list = []
            random_state_list = [random.randint(0,1000)]
            for random_state_manual in random_state_list:
                print('-----------------------------------------------------------------')
                print('random state: ',random_state_manual)

                curr_dir = os.getcwd()
                data = np.load(root_dir)
                X = data['arr_0']
                y = data['arr_1']

                le = preprocessing.LabelEncoder()
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
                
                #create dataloader
                test_dataset2 = mytools.mydata(X_test2,y_test22)
                testloader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, num_workers=0)
                X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.1,random_state=random_state_manual)
                train_dataset = mytools.mydata(X_train,y_train)
                valid_dataset = mytools.mydata(X_valid,y_valid)
                trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                cross_validations = []
                cross_validation_train = []
                F1_Score = []

                # Initialize the teacher model soup
                teacher = teacher_model(ninp=X.shape[2],class_num=class_num,nhid=nhid)
                teacher.load_state_dict(torch.load('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/watkins12/final_checkpoints/checkpoint_0.8686963429328366.pt'))
                # if dataset == 'moby':
                #     souplist = sorted(os.listdir('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/moby'),reverse=True)
                #     souplist = souplist[:5]
                #     souplist = [os.path.join('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/moby',i) for i in souplist]
                # elif dataset == 'watkins':
                #     souplist = sorted(os.listdir('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/watkins/Aug_8'),reverse=True)
                #     souplist = souplist[:5]
                #     souplist = [os.path.join('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/watkins/Aug_8',i) for i in souplist]
                # elif dataset == 'combined':
                #     souplist = sorted(os.listdir('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/combined/Aug_8'),reverse=True)
                #     souplist = souplist[:5]
                #     souplist = [os.path.join('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/combined/Aug_8',i) for i in souplist]
                
                # elif dataset =='watkins12':
                #     souplist = sorted(os.listdir('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/watkins12/final_checkpoints'),reverse=True)
                #     souplist = [os.path.join('/data2/xiangrui_d2/MMC/codes/Models/XCA_cFSMN/checkpoints/xfsmn_checkpoints/8khz/watkins12/final_checkpoints',i) for i in souplist if i.endswith('.pt')]

                
                # teacher_model_soup = model_soup.uniform_soup(teacher_model,souplist)

                # Initialize the student model and prune it
                student_model = nets.student_model(ninp=X.shape[2],class_num=class_num,nhid=nhid)

                parameters_to_prune = ((student_model.cvfsmn, '_memory_weights'),
                                        (student_model.fc1, 'weight'),
                                        (student_model.fc2, 'weight'))
                
                torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=0.2)
                
                total_params =  sum(p.numel() for p in teacher.parameters())
                print(f"{total_params:,} teacher_model totalparameters.")

                total_params =  sum(p.numel() for p in student_model.parameters())
                print(f"{total_params:,} student_model totalparameters.")

                # Initialize the loss functions and optimizer
                loss_function = LabelSmoothingCrossEntropy()
                optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)


                train_time = time.time()
                student_model.train()
                # teacher_model_soup.eval()
                teacher.eval()
                nets.train_cooling_kd_model(teacher,student_model,epoch,trainloader,validloader,optimizer,loss_function,X_train,X_valid,device,verbose=False)
                print('Training time: ',time.time()-train_time)
                train_time_list.append(time.time()-train_time)
                # Evaluate the test set
                y_pred = []
                y_true = []
                compare = []
            
                student_model.load_state_dict(torch.load("./KD_Cool_checkpoint.pt"))
                student_model.eval()
                inference_time = time.time()
                for X, y in testloader2:
                    X = X.to(device)
                    output = student_model(X) # Feed Network
                    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                    y_pred.extend(output) # Save Prediction
                    y = y.data.cpu().numpy()
                    y_true.extend(y) # Save Truth

                for a,b in zip(y_pred,y_true):
                    compare.append(a==b)
                print('testset accuracy: ',sum(compare)/len(y_true)*100)
                print('F1s',precision_recall_fscore_support(torch.tensor(y_pred), torch.tensor(y_true).int(), average='macro'))
                f1 = precision_recall_fscore_support(torch.tensor(y_pred), torch.tensor(y_true).int(), average='macro')
                print(f1[-2])
                if f1[-2] > 0.83:
                    print('save model')
                    f1s.append(f1[:3])
                    shutil.copy('./KD_Cool_checkpoint.pt',f'{checkpoint_dir}/KD_checkpoint_{f1[-2]}.pt')
                # f1s.append(f1[:3])

                print('Inference time: ',time.time()-inference_time)
                infer_time_list.append(time.time()-inference_time)
                # Save the model checkpoint
                                
            # calculate F1 average
        print('--------------------------')
        print('teacher')
        print(teacher)
        print('--------------------------')   
        print('student')
        print(student_model)
        print('--------------------------') 
        f1_ave = np.array(f1s)
        f1_ave = np.mean(f1_ave,axis=0)
        print(f'The average f1 for dataset {dataset} is:',f1_ave)
        print('average training time: ',np.mean(train_time_list))
        print('average inference time: ',np.mean(infer_time_list))
        F1_score = f1_ave[-1]
