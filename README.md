# Human Detector

Human detector is a project intended for human safety and security in an industrial environment.
The neural network detect the presence of a human in an image and can notify to industrial machine to stop works for human safety.

## Run

After clone the repository, you must install torch, torchvision, PIL and numpy.

The command: python train.py run the training step.

It could be run without any argument but you can also specify this parameters:
1. --arch: (architecture) supported architectures are vgg16, vgg19, densenet121, alexnet
2. --learning_rate
3. --hidden_units: the hidden_units numbers
4. --epochs: the epochs for training
5. --drop_prob: the dropout probability

After training complete you can predict the presence of a human in an image using:
python predict.py image_path

The script print the predicted category (human or no human) and the probability


## Evolution

The project is not complete. For complete project it can be useful connect it to a raspberry py with a camera module and run the detection at runtime in images capture
