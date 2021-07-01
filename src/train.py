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

def get_optimizer(model, params):
    """
    Creates an optimizer for the model based on the settings given in params.

    :param model: the model which has to be trained
    :type model: torch.nn.Module
    :param params: the object containing the hyperparameters and other training settings
    :type params: utils.Params
    :return: the optimizer
    """

    optimizer = None
    optimizer_name = params.optimizer
    if optimizer_name == "SGD":
        lr = params.optimizer_settings['learning_rate']
        mom = params.optimizer_settings['momentum']
        wd = params.optimizer_settings['weight_decay']
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=mom,
        #                       weight_decay=wd)
        nesterov = True
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=nesterov)
    elif optimizer_name == "Adam":
        lr = params.optimizer_settings['learning_rate']
        var1 = params.optimizer_settings['beta1']
        var2 = params.optimizer_settings['beta2']
        wd = params.optimizer_settings['weight_decay']
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=[var1, var2],
                               weight_decay=wd)

    return optimizer

project_root_dir = get_project_root()

def get_trained_model(params, model, train_loader, val_loader):
    
    model = model.to(device)
    optimizer = get_optimizer(model, params)
    criterion = nn.CrossEntropyLoss()  # this can be varied, but for common ml classification problems, ce-loss works well
    softmax = nn.Softmax()
    best_model = None
    best_val_acc = 0

    for epoch in range(params.num_epochs):  # loop over the dataset multiple times
        # training epoch
        model.train()
        for batch_idx, (data, target, indexes) in enumerate(train_loader):
            if data.dim() == 3:  # MNIST has 3 dims, we need 4 (batch, channels, pix, pix)
                data = data.unsqueeze(1)
            # get the inputs; data is a list of [inputs, labels]
            data, target = data.type('torch.FloatTensor').to(device), target.type('torch.LongTensor').to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
        # validation step
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch_idx_val, (data, target, indexes) in enumerate(val_loader):
                if data.dim() == 3:  # MNIST has 3 dims, we need 4 (batch, channels, pix, pix)
                    data = data.unsqueeze(1)
                data, target = data.type('torch.FloatTensor').to(device), target.type('torch.LongTensor').to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        val_acc = correct / total
        print('Accuracy of the modelwork on the {} validation images: {:.2%}'.format(total, val_acc))
        if val_acc > best_val_acc:
            best_model = model
            torch.save(model, str(project_root_dir) + str(params.dataset_class_name) + '_' +str(params.id) + '.pt')

    return best_model
    


            
                

    print('Finished Training')

