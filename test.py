

####################################
## 1. Prepare Data for Training
####################################

import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# transforms for image augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

train_path = "DATA/data/train"
val_path = "DATA/data/val"

# write ImageFolder code below
train_data = torchvision.datasets.ImageFolder(
    root = train_path, 
    transform = transform
)
val_data =torchvision.datasets.ImageFolder(
    root = val_path, 
    transform = transform
)
# check the label
# train_data.class_to_idx
# val_data.class_to_idx
print(train_data.class_to_idx)
print(val_data.class_to_idx)


####################################
## 2. Prepare Model
####################################

# assign device cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# import packages for importing models
import torchvision.models as models

# load model and change the # of classes
resnet50 = models.resnet50(pretrained=True)
# 마지막 계층 제외 freeze 
for param in resnet50.parameters():
    param.requires_grad = False
    
num_classes = 1
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
resnet50 = resnet50.to(device)


####################################
## 3. Training
####################################

import wandb
import sklearn
from sklearn import metrics
import datetime
import os
wandb.login()
# use wandb.init
wandb.init(project="mask-classification", name="resnet50-training")
# 모델 파라미터(그래디언트 등) 추적을 위한 .watch 호출
wandb.watch(resnet50)

save_path = "Result/" +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
os.makedirs(save_path, exist_ok=True)


import time
import copy
from torch.autograd import Variable

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels.unsqueeze(1)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.float())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# hyper parameters
num_epochs = 50
learning_rate = 0.01
batch_size = 64

# optimizer, loss, scheduler
optimizer = torch.optim.Adam(resnet50.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()
# LamdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

# data_loader
image_datasets = {'train': train_data, 'val': val_data}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
use_gpu = torch.cuda.is_available()

    
# ROC curve, AUC (Hint: use sklearn or wandb function, using sklearn to extract fpr, tpr will be bonus score)
def visualize_with_wandb(model):
  model.eval()
  all_labels = []
  all_preds = []

  with torch.no_grad():
      for inputs, labels in dataloaders['val']:
          if use_gpu:
              inputs = inputs.cuda()
              labels = labels.cuda()
          outputs = model_ft(inputs)
          preds = torch.sigmoid(outputs)[:, 1]
          all_labels.extend(labels.cpu().numpy())
          all_preds.extend(preds.cpu().numpy())

  fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_labels, all_preds, pos_label=1)
  roc_auc = sklearn.metrics.auc(fpr, tpr)

  print('ROC AUC: {:.4f}'.format(roc_auc))

  # Log ROC curve to wandb
  wandb.log({"roc_curve": wandb.plot.roc_curve(all_labels, all_preds, labels=["Class 0", "Class 1"])})

  # Save the model
  torch.save(model_ft.state_dict(), save_path + "best_model.pth")
  wandb.save(save_path + "best_model.pth")

# perform training and validation
model_ft = train_model(resnet50, criterion, optimizer, scheduler, num_epochs)
visualize_with_wandb(model_ft)
