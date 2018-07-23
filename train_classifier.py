import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
from tqdm import tqdm
from visdom import Visdom
import numpy
import os

parser = argparse.ArgumentParser()
parser.add_argument('train_data')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoches', type=int, default=150)
parser.add_argument('--lr',type=float, default=0.005)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--exp_root', default="./classification")
parser.add_argument('--pretrained')
parser.add_argument('--fine_tune', action='store_true')

opt = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5243, 0.4289, 0.3736],
        std= [0.1202, 0.1094, 0.1154]
    )
])

train_set = torchvision.datasets.ImageFolder(opt.train_data, train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=2)

model = models.resnet34(pretrained=True, num_classes=len(train_set.classes))
if opt.pretrained:
    state_dict = torch.load(opt.pretrained)
    try:
        model.load_state_dict(state_dict)
    except:
        from collections import OrderedDict
        new_state = OrderedDict()
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if not k.startswith('fc'):
                new_state[k] = v
        model_state.update(new_state)
        model.load_state_dict(model_state)

criterion = torch.nn.CrossEntropyLoss()

if opt.fine_tune:
    model.fc.reset_parameters()
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
        {'params': model.relu.parameters()},
        {'params': model.maxpool.parameters()},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters(),},
        {'params': model.avgpool.parameters(),},
        {'params': model.fc.parameters(), 'lr': opt.lr},],
        lr=opt.lr*0.005)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoches, )

batch_size = opt.batch_size

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Plot(object):
    def __init__(self, title, port=8082):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_plot(self, name, xlabel, ylabel):
        win = self.viz.line(
            X=numpy.zeros([1]),
            Y=numpy.zeros([1]),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_plot(self, name, x, y):
        self.viz.line(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name],
            update='append'
        )

plot = Plot("Classification Model" )
plot.register_plot('Loss', 'Iteration', 'Loss')
plot.register_plot('top1', 'Iteration', 'Acc')
plot.register_plot('top5', 'Iteration', 'Acc')

if not os.path.isdir(opt.exp_root):
    os.mkdir(opt.exp_root)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)
for epoch in range(opt.epoches):
    running_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    scheduler.step(epoch)
    for idx, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))

        running_loss.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if idx % opt.log_interval == opt.log_interval -1 :
            total_iterations = epoch * len(train_loader) + idx
            plot.update_plot('Loss', total_iterations, running_loss.avg)
            plot.update_plot('top1', total_iterations, top1.avg)
            plot.update_plot('top5', total_iterations, top5.avg)

    torch.save(model.state_dict(), os.path.join(opt.exp_root, 'model_{}.pth'.format(epoch)))
