import json
import numpy as np
import torch
from torch.autograd import Variable
import argparse
from PIL import Image
from torchvision import models
from classifier import Classifier

default_drop_prob = 0.5

def load_checkpoint():
    nn_filename = 'checkpoint.pth'
    checkpoint = torch.load(nn_filename)
    arch = checkpoint['arch']
    drop_prob = checkpoint['drop_prob'] if hasattr(checkpoint, 'drop_prob') else default_drop_prob
    model = getattr(models, arch)(pretrained=True)
    classifier = Classifier(input_size=checkpoint['input_size'], hidden_size=checkpoint['hidden_size'], output_size=checkpoint['output_size'], drop_prob=drop_prob).classifier
    model.classifier = classifier

    return model

def load_cat_names():
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image, 'r')

    width, height = im.size

    new_width = width

    new_height = height

    if width < height:
        new_width = 256
        new_height = 256 * height / width

    else:
        new_height = 256
        new_width = 256 * width / height

    new_height = int(new_height)
    new_width = int(new_width)

    im.resize((new_width, new_height))

    value = 0.5 * (256 - 224)

    im = im.crop((value, value, 256 - value, 256 - value))

    im = np.array(im) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (im - mean) / std

    image = image.transpose((2, 0, 1))

    return image


def make_prediction(image_path=None):
    if image_path is None:
        return

    enable_gpu = torch.cuda.is_available()

    model = load_checkpoint()

    if enable_gpu:
        model.cuda()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()

    image = Variable(image)

    if enable_gpu:
        image = image.cuda()

    model.eval()

    output = model.forward(image)
    probabilities = torch.exp(output).data
    values, indexes = torch.max(probabilities, 1)
    max_value = values.data.item()
    index = str(indexes.data.item() + 1)

    cat_to_name = load_cat_names()

    print("Predicted cat {} with probability {}".format(cat_to_name[index], max_value))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', default='valid/1/person_001.bmp')

    args = parser.parse_args()

    image_path = args.image_path
    make_prediction(image_path)


main()