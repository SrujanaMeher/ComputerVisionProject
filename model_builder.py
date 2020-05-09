from __future__ import print_function, division

import os
import sys
from collections import defaultdict
from glob import glob
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import bcolz
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets as ds

import files as flz

plt.ion()  # interactive mode
DIR_PATH = os.getcwd()
TEST_PATH = DIR_PATH + '/test/'
MODEL_PATH = DIR_PATH + '/models/'
SUB_PATH = DIR_PATH + '/submissions/'
PRED_PATH = DIR_PATH + '/predictions/'
tst_fpaths, test_fnames = flz.get_paths_to_files(TEST_PATH)


# print(tst_fpaths)
# print(test_fnames)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = DIR_PATH
image_datasets = {x: ds.ImageFolder(os.path.join(data_dir, x),
                                    data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

img_reader = 'pil'
print(TEST_PATH)
tst_dataset: ImageFolder = ds.ImageFolder(TEST_PATH, data_transforms['test'])
tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=4, shuffle=False,
                                         pin_memory=False, num_workers=4)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()



def train_model(model, criterion, optimizer, scheduler, num_epochs=4):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

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
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def train_all():
    ## TRAIN RESNET 18
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    torch.save(model_ft, MODEL_PATH + 'resnet18_t3.pt')

    ## TRAIN RESNET 34
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    torch.save(model_ft, MODEL_PATH + 'resnet34_ft1.pt')

    ## TRAIN RESNET50
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)
    torch.save(model_ft, MODEL_PATH + 'resnet50_ft1.pt')

    ## TRAIN RESNET 101
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)
    torch.save(model_ft, MODEL_PATH + 'resnet101_ft1.pt')

    ## TRAIN RESNET 152
    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=94)
    torch.save(model_ft, MODEL_PATH + 'resnet152_ft1.pt')

    ## TRAIN VGG 19
    model_ft = models.vgg19(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    feature_model = list(model_ft.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, 2))
    model_ft.classifier = nn.Sequential(*feature_model)
    print(num_ftrs)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=50)
    torch.save(model_ft, MODEL_PATH + 'vgg19_ft1.pt')

    ## TRAIN VGG 16
    model_ft = models.vgg16(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    feature_model = list(model_ft.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, 2))
    model_ft.classifier = nn.Sequential(*feature_model)
    print(num_ftrs)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=70)
    torch.save(model_ft, MODEL_PATH + 'vgg16_ft1.pt')


# Code for generating predictions source: https://github.com/bfortuner/pytorch-kaggle-starter
# Original Author: Brendan Fortuner
def predict_batch(net, inputs):
    v = Variable(inputs.cuda(), volatile=True)
    return net(v).data.cpu().numpy()


def get_probabilities(model, loader):
    model.eval()
    return np.vstack(predict_batch(model, data[0]) for data in loader)


def get_prediction_fpath(basename, dset):
    fname = '{:s}_{:s}'.format(basename, dset + '.bc')
    return os.path.join(PRED_PATH, fname)


def save_or_append_pred_to_file(fpath, pred_arr, meta_dict=None):
    if os.path.exists(fpath):
        return append_pred_to_file(fpath, pred_arr, meta_dict)
    else:
        return save_pred(fpath, pred_arr, meta_dict)


def save_pred(fpath, pred_arr, meta_dict=None):
    bc = bcolz.carray(pred_arr, mode='w', rootdir=fpath,
                      cparams=bcolz.cparams(clevel=9, cname='lz4'))
    if meta_dict is not None:
        bc.attrs['meta'] = meta_dict
    bc.flush()
    return bc


def append_pred_to_file(fpath, pred_arr, meta_dict=None):
    bc_arr = bcolz.open(rootdir=fpath)
    bc_arr.append(pred_arr)
    if meta_dict is not None:
        bc_arr.attrs['meta'] = meta_dict
    bc_arr.flush()
    return bc_arr


def get_sub_path_from_pred_path(pred_fpath):
    sub_fname = os.path.basename(pred_fpath).rstrip(
        '.bc') + '.csv'
    sub_fpath = os.path.join(SUB_PATH, sub_fname)
    return sub_fpath


def get_fnames_from_fpaths(fpaths):
    fnames = []
    for f in fpaths:
        if isinstance(f, tuple):
            f = f[0]
        fnames.append(os.path.basename(f))
    return fnames


def make_preds_submission(sub_fpath, ids, preds, header):
    preds = [' '.join(map(str, p.tolist())) for p in preds]
    write_preds_to_file(sub_fpath, ids, preds, header)


def write_preds_to_file(fpath, ids, preds, header):
    ids = np.array(ids).T
    preds = np.array(preds).T
    submission = np.stack([ids, preds], axis=1)
    np.savetxt(fpath, submission, fmt='%s', delimiter=',',
               header=header, comments='')


def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    if method == "average":
        scores = defaultdict(float)
    with open(loc_outfile, "w") as outfile:
        for i, glob_file in enumerate(glob(glob_files)):
            print("parsing: {}".format(glob_file))
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate(lines):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    if scores[(e, row[0])] == 0:
                        scores[(e, row[0])] = 1
                    scores[(e, row[0])] *= float(row[1])
        for j, k in sorted(scores):
            outfile.write("%s,%f\n" % (k, math.pow(scores[(j, k)], 1 / (i + 1))))
        print("wrote to {}".format(loc_outfile))


def ensemble_predictions():
    for filename in os.listdir(MODEL_PATH):
        model_ft = torch.load(MODEL_PATH + filename)
        tst_probs = get_probabilities(model_ft, tst_loader)
        (prefix, sep, suffix) = filename.rpartition('.')
        filename = 'm_' + prefix
        pred_fpath = get_prediction_fpath(basename=filename, dset='_t1')
        _ = save_or_append_pred_to_file(pred_fpath, tst_probs)
        ub_fpath = get_sub_path_from_pred_path(pred_fpath)
        fnames = get_fnames_from_fpaths(tst_fpaths)
        sub_ids = [f.split('.')[0] for f in fnames]
        tst_probs = np.clip(tst_probs, .03, .97)
        make_preds_submission(ub_fpath, sub_ids, np.expand_dims(tst_probs[:, 1], 1), 'Id,Probability')

    glob_files = SUB_PATH + 'm*.csv'
    loc_outfile = SUB_PATH + 'geomean.csv'
    kaggle_bag(glob_files, loc_outfile)


if __name__ == '__main__':
    if (sys.argv[1] == "-predict"):
        ensemble_predictions()
    elif (sys.argv[1] == "-trainall"):
        train_all()
    else:
        print("Rerun this program with '-trainall' or '-predict' to retrain networks or generate predictions")
