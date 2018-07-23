import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys
import math
import torch.nn as nn
import torch.nn.functional as F
from knn_search_util import search_index_pytorch
import faiss


def init_dataset(dataset_root, batch_size):
    '''
    Initialize the datasets, samplers and dataloaders
    '''

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5243, 0.4289, 0.3736],
            std= [0.1202, 0.1094, 0.1154]
        )
    ])


    train_dataset = ImageFolder(dataset_root,
                                train_transform)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=False)

    return train_dataset, tr_dataloader


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def init_protonet():
    '''
    Initialize the ProtoNet
    '''
    # model = ProtoNet().to(device)
    modules = list(models.resnet34(pretrained=False).children())[:-1]
    modules.append(Flatten())
    model = nn.Sequential(*modules)
    return model

class FeatModel(nn.Module):
    def __init__(self, model):
        super(FeatModel, self).__init__()
        self.features = nn.Sequential(
                *list(model.children())[:-1]
                )
        self.features.add_module('Flatten', Flatten())

    def forward(self, x):
        x = self.features(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class DistillModel(nn.Module):
    def __init__(self, model, num_classes):
        super(DistillModel, self).__init__()
        self.feat = nn.Sequential(
                *list(model.children())[:-1]
                )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        feature = self.feat(x)
        logits = self.classifier(
                feature.view(feature.size(0), -1))
        return feature, logits

def main(args):

    device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'
    with torch.no_grad():
        trainset, train_loader = init_dataset(args.train_data_dir, args.batch_size)
        testset, test_loader = init_dataset(args.test_data_dir, args.batch_size)

        model_list = []
        for ckpt in args.model: 
            if args.use_proto:
                model = init_protonet()
            elif args.use_class:
                class_model = models.resnet34(pretrained=False, num_classes=1000)
                class_model.load_state_dict(torch.load(ckpt))
                model = FeatModel(class_model )
            elif args.use_distill:
                class_model = models.resnet34(pretrained=False, num_classes=1000)
                model = DistillModel(class_model, 1000)
                model.load_state_dict(torch.load(ckpt))
                model = FeatModel(model)
            else:
                class_model = models.resnet34(pretrained=False)
                model = FeatModel(class_model)
                model.load_state_dict(torch.load(ckpt))
                
            model.to(device)
            model_list.append(model)



        feat_novel = torch.zeros((len(trainset), 512))
        label_novel = torch.zeros((len(trainset)))

        feat_query = torch.zeros((len(testset), 512))
        label_query = torch.zeros((len(testset)))

        print('Runing forward on noval images')
        # tr_iter = iter(train_loader)
        for idx, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_outputs = [model(x).unsqueeze(0) for model in model_list]
            model_output, _ = torch.max(torch.cat(model_outputs), 0)
            start_idx = idx*args.batch_size
            end_idx = min((idx+1)*args.batch_size, len(trainset))
            feat_novel[start_idx: end_idx, :] = model_output
            label_novel[start_idx: end_idx] = y

        print('Runing forward on query images')
        for idx, batch in enumerate(tqdm(test_loader)):
            x, y = batch
            x, y = x.cuda(), y.cuda()
            model_output = model(x)
            start_idx = idx*args.batch_size
            end_idx = min((idx+1)*args.batch_size, len(testset))
            feat_query[start_idx: end_idx, :] = model_output
            label_query[start_idx: end_idx] = y

        labels0 = label_novel.data.cpu().numpy()
        labels1 = label_query.data.cpu().numpy()
        same = labels0 == labels1[:, np.newaxis]
        r, c = np.where(same)

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, 512)
        index.add(feat_novel.data.cpu().numpy())

        #  top 5 precision
        k5 = 5                          # we want to see 5 nearest neighbors
        D5, I5 = search_index_pytorch(index, feat_query, k5)
        prec5 = (np.isin(c.reshape(-1, 1), I5[r])).sum() / c.shape[0]

        # top 1 acc
        k1 = 1
        D1, I1 = search_index_pytorch(index, feat_query, k1)
        prec1 = (c.reshape(-1, 1) == I1[r]).sum().item() / c.shape[0]

        print("top 5 precision {}".format(prec5))
        print("top 1 precision {}".format(prec1))
        # print("recall {}".format(c.shape[0]/2000))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('test_data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, nargs='+',
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--use_proto', action='store_true')
    parser.add_argument('--use_class', action='store_true')
    parser.add_argument('--use_distill', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
