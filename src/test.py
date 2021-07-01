"""
Simple training procedure, mainly from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import sys
def get_project_root():
    return Path(__file__).parent.parent
sys.path.append(get_project_root())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


project_root_dir = get_project_root()

def evaluate(params, model, test_loader):

    classes = params.labels
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model = model.eval().to(device)
    total, correct = 0, 0
    with torch.no_grad():
        for batch_idx_val, (data, target, indexes) in enumerate(test_loader):
            if data.dim() == 3:  # MNIST has 3 dims, we need 4 (batch, channels, pix, pix)
                data = data.unsqueeze(1)
            data, target = data.type('torch.FloatTensor').to(device), target.type('torch.LongTensor').to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            for label, prediction in zip(target, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    test_acc = correct / total
    print('Accuracy of the modelwork on the {} validation images: {:.2%}'.format(total, test_acc))
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))

    return test_acc
    

