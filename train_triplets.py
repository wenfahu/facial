import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from BalancedSampler import BalancedBatchSampler
from triplet_loss import HardTripletLoss, SoftHardTripletLoss, SoftHingeTripletLoss
import argparse
from tqdm import tqdm
from visdom import Visdom
import numpy
import os
from plot import LinePlot

from comet_ml import Experiment
experiment  = Experiment(api_key="c3UWUJzB3uF5NMnnhu4xpDymo")

parser = argparse.ArgumentParser()
parser.add_argument('train_data')
parser.add_argument('--num_classes', type=int, default=64)
parser.add_argument('--num_samples_per_class', type=int, default=2)
parser.add_argument('--epoches', type=int, default=150)
parser.add_argument('--lr',type=float, default=0.005)
parser.add_argument('--pretrained')
parser.add_argument('--resume')
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--exp_root', default="./triplet")
parser.add_argument('--margin', type=float, default=0.)
parser.add_argument('--hard_margin', action='store_true')
parser.add_argument('--soft_margin', action='store_true')
parser.add_argument('--soft_hinge', action='store_true')

opt = parser.parse_args()

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5243, 0.4289, 0.3736],
        std= [0.1202, 0.1094, 0.1154]
    )
])

class FeatModel(nn.Module):
    def __init__(self, model):
        super(FeatModel, self).__init__()
        self.features = nn.Sequential(
                *list(model.children())[:-1]
                )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x

train_set = torchvision.datasets.ImageFolder(opt.train_data, train_transform)
train_sampler = BalancedBatchSampler(train_set, opt.num_classes, opt.num_samples_per_class)
train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler,)

model = models.resnet34(pretrained=False, num_classes=8631)
if opt.pretrained:
    model.load_state_dict(torch.load(opt.pretrained))
model = FeatModel(model)
if opt.resume:
    model.load_state_dict(torch.load(opt.resume))
if opt.soft_hinge:
    criterion = SoftHingeTripletLoss(margin=opt.margin )
    plot = LinePlot("Triplet Model (soft hinge)")
elif opt.hard_margin:
    criterion = HardTripletLoss(margin=opt.margin, hardest=True)
    plot = LinePlot("Triplet Model (hard margin)")
elif opt.soft_margin:
    criterion = SoftHardTripletLoss(hardest=True)
    plot = LinePlot("Triplet Model (soft margin)")


optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoches, )


plot.register_plot('Loss', 'Iteration', 'Loss')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)
for epoch in range(opt.epoches):
    running_loss = 0.0
    scheduler.step(epoch)
    for idx, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if idx % opt.log_interval == opt.log_interval -1 :
            total_iterations = epoch * len(train_loader) + idx
            avg_loss = running_loss / idx
            experiment.log_metric("triplet loss", avg_loss)
            plot.update_plot('Loss', total_iterations, avg_loss)

    if not os.path.isdir(opt.exp_root):
        os.makedirs(opt.exp_root)
    torch.save(model.state_dict(), os.path.join(opt.exp_root, 'model.pth'))
