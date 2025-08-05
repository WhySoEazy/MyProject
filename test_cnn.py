from argparse import ArgumentParser
import torch
from model import SimpleCNN
import cv2
import numpy as np
import torch.nn as nn

def get_args():
    parse = ArgumentParser(description="CNN inference")

    parse.add_argument("--image_size" , type=int , default=224)
    parse.add_argument("--checkpoint" , type=str , default="trained_model/best_cnn.pt")
    parse.add_argument("--image_path" , type=str , default=None)

    arg = parse.parse_args()

    return arg

if __name__ == "__main__":
    arg = get_args()

    categories = ['cats' , 'dogs']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SimpleCNN(num_classes=2).to(device)

    if arg.checkpoint:
        checkpoint = torch.load(arg.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found!")
        exit(0)

    model.eval()

    image = cv2.imread(arg.image_path)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (arg.image_size , arg.image_size))
    image = np.transpose(image , (2 , 0 , 1))/255.0
    image = image[None , : , : , :]
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        output = model(image)
        probs = softmax(output)

    max_prob = torch.max(probs)
    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    print("This image is about {} with {} probability".format(predicted_class , max_prob))