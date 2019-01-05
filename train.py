import argparse
import os
from classifier import Classifier
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms

default_arch = "vgg16"
default_learning_rate = 0.05
default_input_size = 25088
default_hidden_size = 512
default_output_size = 2
default_epochs = 20
default_drop_prob = 0.5

available_archs = {"vgg16", "vgg19", "densenet121", "alexnet"}

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def train(arch=default_arch, learning_rate=default_learning_rate, input_size=default_input_size, hidden_size=default_hidden_size, output_size=default_output_size, epochs=default_epochs, drop_prob=default_drop_prob):
    print("Train function")

    data_dir = '.'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=20)

    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size, drop_prob=drop_prob).classifier
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    steps = 0
    training_loss = 0
    print_every = 20

    enable_gpu = torch.cuda.is_available()

    if enable_gpu:
        model.cuda()

    for e in range(epochs):
        model.train()

        for ii, (data, target) in enumerate(trainloader):
            steps += ii

            data, target = Variable(data), Variable(target)

            if enable_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            output = model.forward(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            training_loss += loss.data.item()

            if steps % print_every == 0:

                model.eval()

                accuracy = 0
                valid_loss = 0

                for jj, (inputs, label) in enumerate(validloader):

                    inputs, label = Variable(inputs), Variable(label)

                    if enable_gpu:
                        inputs, label = inputs.cuda(), label.cuda()

                    output = model.forward(inputs)
                    loss = criterion(output, label)

                    valid_loss += loss.data.item()

                    ps = torch.exp(output).data

                    # Class with highest probability is our predicted class, compare with true label
                    # Gives index of the class with highest probability, max(ps)
                    equality = (label.data == ps.max(1)[1])

                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}   ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}..".format(100 * accuracy / len(validloader)))

                training_loss = 0

                # Model in training mode
                model.train()

    return model, classifier, optimizer


def save_checkpoint(model=None, classifier=None, optimizer=None, arch=default_arch, input_size=default_input_size, hidden_size=default_hidden_size, output_size=default_output_size, learning_rate=default_learning_rate):
    nn_filename = 'checkpoint.pth'
    checkpoint = {
        'state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'arch': arch
    }
    if os.path.exists(nn_filename):
        try:
            os.remove(nn_filename)
        except OSError:
            print("No such file")

    torch.save(checkpoint, nn_filename)

    if os.path.exists(nn_filename):
        print("Checkpoint saved")


def main():
    print("Start main")

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=available_archs)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_units", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--drop_prob", type=float)

    args = parser.parse_args()

    arch = args.arch if args.arch is not None else default_arch
    learning_rate = args.learning_rate if args.learning_rate is not None else default_learning_rate
    hidden_size = args.hidden_units if args.hidden_units is not None else default_hidden_size
    epochs = args.epochs if args.epochs is not None else default_epochs
    drop_prob = args.drop_prob if args.drop_prob is not None else default_drop_prob

    print("Start train")

    model, classifier, optimizer = train(arch, learning_rate, default_input_size, hidden_size, default_output_size, epochs, drop_prob)

    save_checkpoint(model, classifier, optimizer, arch, default_input_size, hidden_size, default_output_size, learning_rate)


main()