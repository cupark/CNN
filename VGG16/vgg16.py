import torch
from torch.utils.data import DataLoader 
import torchvision
from torchvision import transforms as tf, datasets #torchvision모듈 내 transforms, datasets함수 임포트

import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

from torch import optim
from torch.optim.lr_scheduler import StepLR


import matplotlib.pyplot as plt
import numpy as np

import json
import os
import time
import copy

Path_MNIST = '../VGG16/data/MNIST/'
Path_STL10 = '../VGG16/data/STL10/'

#HyperParam
IMPageNet_BATCH_SIZE = 16
STL10_BATCH_SIZE = 4

CONV_KERNEL_SIZE = 3
c_stride = 1
c_padding = 1

MAXP_KERNEL_SIZE = 2
mp_stride = 2

lr = 0.01
step_size=30
gamma= 0.1


transform = tf.Compose([
    tf.Resize((224, 224)),
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
])

train_dataset = datasets.MNIST(root =Path_MNIST,
                               train = True,
                               download = True,
                               transform = tf.ToTensor( ))


test_dataset = datasets.MNIST(root =Path_MNIST,
                               train = False,
                               download = True,                            
                               transform = tf.ToTensor( ))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size =IMPageNet_BATCH_SIZE,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = IMPageNet_BATCH_SIZE,
                                           shuffle = False)

for (X_train, y_train) in train_loader:
     X_train.size()
     X_train.type()
     y_train.size()
     y_train.type()
    
print('X_train:', X_train.size( ), '\ntype:', X_train.type( ))
print('y_train:', y_train.size( ), '\ntype:', y_train.type( ))
     
pltsize = 1
plt.figure(figsize=(20 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.title("NUM : " + str(y_train[i].item()))
    plt.imshow(X_train[i, :, :, :].numpy( ).reshape(28,28))
     
     
STL_train_dataset = datasets.STL10(root =Path_STL10,
                               split='train',
                               download = True,
                               transform = tf.ToTensor())


STL_test_dataset = datasets.STL10(root =Path_STL10,
                               split='test',
                               download = True,                            
                               transform = tf.ToTensor())

img, _ = STL_train_dataset[1]
print("stl img shape: ", img.shape) # size = [3, 96, 96]
print("len stl_train", len(STL_train_dataset)) # 학습용 데이터 5000개
print("len stl_test", len(STL_test_dataset))   # 테스트 데이터 8000개

STL_train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in STL_train_dataset]
STL_train_stdRGB  = [np.std(x.numpy(), axis=(1,2))  for x, _ in STL_train_dataset]

STL_train_meanR = np.mean([m[0] for m in STL_train_meanRGB])
STL_train_meanG = np.mean([m[1] for m in STL_train_meanRGB])
STL_train_meanB = np.mean([m[2] for m in STL_train_meanRGB])
STL_train_stdR  = np.std([s[0] for s in STL_train_stdRGB])
STL_train_stdG  = np.std([s[1] for s in STL_train_stdRGB])
STL_train_stdB  = np.std([s[2] for s in STL_train_stdRGB])

print("STL_train_meanR: ",STL_train_meanR) #STL_train_meanR:  0.4467106
print("STL_train_meanG: ",STL_train_meanG) #STL_train_meanG:  0.43980986
print("STL_train_meanB: ",STL_train_meanB) #STL_train_meanB:  0.40664646
print("STL_train_stdR: ", STL_train_stdR)  #STL_train_stdR:   0.061958
print("STL_train_stdG: ", STL_train_stdG)  #STL_train_stdG:   0.06189533
print("STL_train_stdB: ", STL_train_stdB)  #STL_train_stdB:   0.068922155

STL_test_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in STL_test_dataset]
STL_test_stdRGB  = [np.std(x.numpy(), axis=(1,2))  for x, _ in STL_test_dataset]

STL_test_meanR = np.mean([m[0] for m in STL_test_meanRGB])
STL_test_meanG = np.mean([m[1] for m in STL_test_meanRGB])
STL_test_meanB = np.mean([m[2] for m in STL_test_meanRGB])
STL_test_stdR  = np.std( [s[0] for s in STL_test_stdRGB])
STL_test_stdG  = np.std( [s[1] for s in STL_test_stdRGB])
STL_test_stdB  = np.std( [s[2] for s in STL_test_stdRGB])

print("STL_test_meanR: ",STL_test_meanR) #STL_test_meanR:  0.44723064
print("STL_test_meanG: ",STL_test_meanG) #STL_test_meanG:  0.4396425
print("STL_test_meanB: ",STL_test_meanB) #STL_test_meanB:  0.40495726
print("STL_test_stdR: ", STL_test_stdR)  #STL_test_stdR:   0.0617221
print("STL_test_stdG: ", STL_test_stdG)  #STL_test_stdG:   0.06181906
print("STL_test_stdB: ", STL_test_stdB)  #STL_test_stdB:   0.06902452

STL_train_transformer = tf.Compose([
    tf.Resize(224),
    #tf.CenterCrop(224),
    #tf.Lambda(lambda crops: torch.stack([tf.ToTensor()(crop) for crop in crops])),
    tf.ToTensor(),
    tf.Normalize([STL_train_meanR, STL_train_meanG, STL_train_meanB], [STL_train_stdR, STL_train_stdG, STL_train_stdB]),
])
STL_train_dataset.transform = STL_train_transformer
STL_test_dataset.transform = STL_train_transformer

def show_img(imgs, y=None, color=True):
    for i, img in enumerate(imgs):
        np_img = img.numpy()
        np_img_tr = np.transpose(np_img, (1, 2, 0))
        plt.subplot(1, imgs.shape[0], i+1)
        plt.imshow(np_img_tr)
        
    if y is not None:
        plt.title('labels: ' + str(y))

np.random.seed(0)
torch.manual_seed(0)

rnd_idx = int(np.random.randint(0, len(STL_train_dataset) ,1))
img, label = STL_train_dataset[rnd_idx]
print("image shape : ", img.shape)
print("label : ", label)
print("image index: ", rnd_idx)

plt.figure(figsize=(20,20))
#show_img(img)
        
STL_train_dl = DataLoader(STL_train_dataset, batch_size=STL10_BATCH_SIZE, shuffle=True)
STL_test_dl = DataLoader(STL_test_dataset, batch_size=STL10_BATCH_SIZE, shuffle=True)        

VGG_Type = {
    'VGG11': [64, 'MP', 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    'VGG13': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    'VGG16': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP'],
    'VGG19': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512,  'MP', 512, 512, 512, 512, 'MP']
}

class VGGnet(nn.Module):
    def __init__(self, model, in_ch=3, cls = 10, init_w=True):
        super(VGGnet, self).__init__()
        self.in_ch = in_ch
        self.conv_layers = self.create_conv_laters(VGG_Type[model])
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, cls),
        )
        if init_w: 
            self.init_weight()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512*7*7)
        x = self.fc(x)
        return x
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode ='fan_out', nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def create_conv_laters(self, architecture):
        layers = []
        in_chs = self.in_ch
        for x in architecture:
            if type(x) == int:
                out_ch = x             
                layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_ch, kernel_size=(CONV_KERNEL_SIZE,CONV_KERNEL_SIZE), stride=(c_stride,c_stride), padding=(c_padding,c_padding)),
                                     nn.BatchNorm2d(x), nn.ReLU()]
                in_chs = x 
                
        
            elif x == 'MP':
                layers += [nn.MaxPool2d(kernel_size=(MAXP_KERNEL_SIZE,MAXP_KERNEL_SIZE), stride=(mp_stride,mp_stride))]
            
        return nn.Sequential(*layers)
        
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()

model = VGGnet('VGG16', in_ch=3, cls=10, init_w=True).to(device)

print(model)

summary(model, input_size=(3, 224, 224), device=device.type)
        
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model.parameters(), lr = lr)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
current_lr = get_lr(opt)
print('current lr : ', current_lr)

lr_scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

def metrics_batch(output, target):
    pred = output.argmax(dim =1, keepdim = True)
    corrects =pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metrics_b = metrics_batch(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metrics_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check = False, opt = None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
                
        #bs, ncrops, c, h, w = xb.size()
        bs, c, h, w = xb.size()
        output_ = model(xb.view(-1, c, h, w))
        #output = output_.view(bs, ncrops, -1).mean(1)
        #output = output_.view(bs, -1).mean(1)
        
        loss_b, metric_b = loss_batch(loss_func, output_, yb, opt)
        
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break
        
    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    
    return loss, metric
        
def train_val(model, params):
    # extract model parameters
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    STL_train_dl=params["STL_train_dl"]
    STL_test_dl=params["STL_test_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    # history of loss values in each epoch
    loss_history={
        "train": [],
        "val": [],
    }
    
    # histroy of metric values in each epoch
    metric_history={
        "train": [],
        "val": [],
    }
    
    # 가중치를 저장할 때, 코랩 GPU 오류나서 생략했습니다.
    # a deep copy of weights for the best performing model
    # best_model_wts = copy.deepcopy(model.state_dict())
    
    # initialize best loss to a large value
    best_loss=float('inf')
    
    # main loop
    for epoch in range(num_epochs):
        # check 1 epoch start time
        start_time = time.time()

        # get current learning rate
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        # train model on training dataset
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,STL_train_dl,sanity_check,opt)

        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # evaluate model on validation dataset    
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,STL_test_dl,sanity_check)
        
       
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # # store weights into a local file
            # torch.save(model.state_dict(), path2weights)
            # print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # learning rate schedule
        lr_scheduler.step()

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f, time: %.4f s" %(train_loss,val_loss,100*val_metric, time.time()-start_time))
        print("-"*10) 

    ## load best model weights
    # model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

# definc the training parameters
params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'STL_train_dl':STL_train_dl,
    'STL_test_dl':STL_test_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except :
        return 'error'
createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)