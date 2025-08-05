import torch
import numpy as np
from torch.utils.data import Dataset , DataLoader
import os
import cv2
from PIL import Image
from torchvision.transforms import ToTensor , Resize , Compose

class DogsCatsDataset(Dataset):
    def __init__(self , root , train=True , transform = ToTensor()):

        self.categories = ['cats' , 'dogs']
        self.images_path = []
        self.labels = []
        self.transform = transform

        if train==True:
            folder_path = os.path.join(root , "train")
        else:
            folder_path = os.path.join(root , "test")

        for i , category in enumerate(self.categories):
            category_path = os.path.join(folder_path , category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path , image_name)
                self.images_path.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image , label

if __name__ == "__main__":
    root="small_dog_cat_dataset"
    tramsform = Compose([
        Resize((224,224)),
        ToTensor()
    ])
    training_data = DogsCatsDataset(root="small_dog_cat_dataset" , train=True , transform = tramsform)
    # image , label = training_data.__getitem__(1234)
    training_dataloader = DataLoader(
        dataset=training_data,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )
    for images , labels in training_dataloader:
        print(images.shape)
        print(labels)