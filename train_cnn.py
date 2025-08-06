from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from dogs_cats_dataset import DogsCatsDataset
from torchvision.transforms import ToTensor , Resize , Compose , Normalize , ColorJitter , RandomAffine
from model import SimpleCNN
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import cv2

def get_args():
    parse = ArgumentParser(description="CNN Training")

    parse.add_argument("--root", 
                       type=str, 
                       default="./small_dog_cat_dataset")
    
    parse.add_argument("--epochs",
                       type=int,
                       default=100)
    
    parse.add_argument("--batchs",
                       type=int,
                       default=8)
    
    parse.add_argument("--logging",
                       type=str,
                       default="tensorboard")
    
    parse.add_argument("--trained_model",
                       type=str,
                       default="trained_model")
    
    parse.add_argument("--checkpoint",
                       type=str,
                       default=None)

    parse.add_argument("--image_size",
                       type=int,
                       default=224)
    
    args = parse.parse_args()

    return args

def plot_condusion_matrix(writer , cm , class_names , epoch):
    figure = plt.figure(figsize=(20,20))
    plt.imshow(cm , interpolation="nearest" , cmap="Wistia")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks , class_names , rotation=45)
    plt.yticks(tick_marks , class_names)
    
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] , decimals=2)
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > threshold else "black"
            plt.text(j , i, cm[i,j], horizontalalignment = "center" , color=color)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    writer.add_figure("Confusion matrix" , figure , epoch)    

if __name__ == "__main__":

    arg = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_epochs = arg.epochs
    batch_size = arg.batchs

    train_transform = Compose([
        RandomAffine(
            degrees=(-5 , 5),
            translate=(0.05 , 0.05),
            scale=(0.85 , 1.15),
            shear=5
        ),
        ColorJitter(
            brightness=0.125,  
            contrast=0.5,   
            saturation=0.25,  
            hue=0.5
        ),
        Resize((arg.image_size , arg.image_size)),
        ToTensor()
    ])

    test_transform = Compose([
        Resize((arg.image_size , arg.image_size)),
        ToTensor()
    ])

    training_data = DogsCatsDataset(root=arg.root , train=True , transform = train_transform)

    training_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    testing_data = DogsCatsDataset(root=arg.root , train=False , transform = test_transform)

    testing_dataloader = DataLoader(
        dataset=testing_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    if os.path.isdir(arg.logging):
        shutil.rmtree(arg.logging , ignore_errors=True)

    if not os.path.isdir(arg.trained_model):
            os.mkdir(arg.trained_model)

    writer = SummaryWriter(arg.logging)
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters() , lr=0.001 , momentum=0.9)
    num_iters = len(training_dataloader)
    summary(model , (3 , arg.image_size , arg.image_size))

    if arg.checkpoint:
        checkpoint = torch.load(arg.checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch , arg.epochs + start_epoch): 
        model.train()
        progress_bar = tqdm(training_dataloader)
        for iter , (images , labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            #forward pass
            outputs = model(images)
            loss = criterion(outputs , labels)
            progress_bar.set_description("Epoch {}/{} , Iteration {}/{} , Loss {:.3f}".format(epoch + 1 , arg.epochs + start_epoch , iter + 1 , num_iters , loss))
            writer.add_scalar("Train/Loss" , loss , epoch*num_iters+iter)
            #backward and optimize
            optimizer.zero_grad() #k cần lưu trữ gradient
            loss.backward() #tính gradient
            optimizer.step() #update parameters

        model.eval()

        all_predictions = []
        all_labels = []

        for iter , (images , labels) in enumerate(testing_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu() , dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(outputs , labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_condusion_matrix(writer, 
                              confusion_matrix(all_labels, all_predictions), 
                              class_names=testing_data.categories, 
                              epoch=epoch)
        accuracy = accuracy_score(all_labels , all_predictions)
        print("Epoch: {} , Accuracy Score: {}".format(epoch+1 , accuracy))
        writer.add_scalar("Validation/Accuracy" , accuracy , epoch)

        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(arg.trained_model))
            best_acc = accuracy

        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(arg.trained_model))
        
        print("Best accuracy: {}".format(best_acc))