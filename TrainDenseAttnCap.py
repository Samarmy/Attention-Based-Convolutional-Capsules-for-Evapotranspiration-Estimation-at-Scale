import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, interpolate
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import glob
import math
from functools import partial
from torchvision import transforms, utils
import random
import os
from datetime import datetime
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, interpolate
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import glob
import math
from functools import partial
from torchvision import transforms, utils
import random
import os
from datetime import datetime
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
from collections import OrderedDict
from MLBop import CNN, Encoder
import socket
import time

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        
    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels,self.out_channels)
        self.relu1 = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4,self.out_channels,kernel_size=1,stride=1,padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask

class EvapoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, split=1):
        
        
        self.file_names = []
        for i in range(1, 6):
            if (train and (i != split)):
                self.file_names = self.file_names + glob.glob("split_data/" + str(split) + "/*")
            if ((train is False) and (i == split)):
                self.file_names = self.file_names + glob.glob("split_data/" + str(split) + "/*")
              
        self.vegs = ['DBF', 'OSH', 'GRA', 'WET', 'SAV', 'ENF', 'MF', 'WSA', 'CRO']
        self.clims = ['Dfb', 'Bwk', 'Cfa', 'Cwa', 'Dwb', 'Dfc', 'Dfa', 'Bsk', 'Csa', 'Bsh']
        
        print("Dataset Length " + str(len(self.file_names)))
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        lon, lat, elev, veg, clim, geohash, year, month, day, cloud_coverage, pixel_coverage, true_et, pred_et = self.file_names[idx].split("_")[-13:]
        img = torch.from_numpy(np.load(self.file_names[idx]).astype(float))
        et = float(self.file_names[idx].split("_")[-1].replace(".npy", ""))
        date = "_".join(self.file_names[idx].split("_")[-7:-4])
        lat = float(self.file_names[idx].split("_")[-12])
        lon = float(self.file_names[idx].split("_")[-13])
        elev = np.array([float(self.file_names[idx].split("_")[-11])/8848.0])
        veg = torch.nn.functional.one_hot(torch.tensor(self.vegs.index(self.file_names[idx].split("_")[-10])), num_classes=len(self.vegs))
        clim = torch.nn.functional.one_hot(torch.tensor(self.clims.index(self.file_names[idx].split("_")[-9])), num_classes=len(self.clims))
        year = self.file_names[idx].split("_")[-7]
        month = self.file_names[idx].split("_")[-6]
        day = self.file_names[idx].split("_")[-5]
        
        date_time_obj = datetime.strptime(year + '_' + month + '_' + day, '%Y_%m_%d')
        day_of_year = date_time_obj.timetuple().tm_yday
        day_sin = torch.tensor([np.sin(2 * np.pi * day_of_year/364.0)])
        day_cos = torch.tensor([np.cos(2 * np.pi * day_of_year/364.0)])
        
        x_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.cos(np.deg2rad(lon))])
        y_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.sin(np.deg2rad(lon))])
        z_coord = torch.tensor([np.cos(math.pi/2-np.deg2rad(lat))])
        
        img = interpolate(img , size=32)[0]
        
        if img[20].mean() < 0:
            lon_img = img[20].clone()
            lat_img = img[19].clone()
        else:
            lat_img = img[20].clone()
            lon_img = img[19].clone()
            
        img[19] = lon_img
        img[20] = lat_img

        return img, et, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, date, lon, lat, self.file_names[idx]         

class convolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=1, padding=2,
                 kernel=5, num_routes=3, nonlinearity='sqaush', batch_norm=False, dynamic_routing='local', cuda=False):
        super(convolutionalCapsule, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(in_capsules*out_capsules*out_channels)
        self.conv2d = nn.Conv2d(kernel_size=(kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels*out_capsules)
        self.dynamic_routing = dynamic_routing
        self.cuda = cuda
        self.SAM1 = SAM(self.in_channels,self.in_channels,1)

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height = x.size(3), x.size(4)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height)
        u_hat = self.conv2d(x) * self.SAM1(x)

        out_width, out_height = u_hat.size(2), u_hat.size(3)

        # batch norm layer
        if self.batch_norm:
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = u_hat.view(batch_size, self.in_capsules * self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = self.bn(u_hat)
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules*self.out_channels, out_width, out_height)
            u_hat = u_hat.permute(0,1,3,4,2).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)

        else:
            u_hat = u_hat.permute(0,2,3,1).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules*self.out_channels)
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)


        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, self.out_capsules))
        if self.cuda:
            b_ij = b_ij.cuda()
        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(5)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)


            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 1, 5)
                temp2 = v_j.unsqueeze(5)
                a_ij = torch.matmul(temp, temp2).squeeze(5) # dot product here
                a_ij = a_ij.permute(0, 4, 1, 2, 3)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 3, 4, 1, 2).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
class _DenseLayer(nn.Sequential):

    def __init__(self, num_caps, num_input_features, growth_rate, bn_size, drop_rate, actvec_size):
        super().__init__()
        self.concap1 = convolutionalCapsule(in_capsules=num_caps, out_capsules=num_caps, in_channels=num_input_features,
                                  out_channels=bn_size * growth_rate,
                                  stride=1, padding=1, kernel=3, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True)
        
        self.concap2 = convolutionalCapsule(in_capsules=num_caps, out_capsules=num_caps, in_channels=bn_size * growth_rate,
                                  out_channels=growth_rate,
                                  stride=1, padding=1, kernel=3, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True)
        

    def forward(self, x):
        new_features = self.concap1(x)
        new_features = self.concap2(new_features)
        return torch.cat([x, new_features], 2)
            
class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_caps, num_input_features, bn_size, growth_rate,
                 drop_rate, actvec_size):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_caps, num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, actvec_size)
            self.add_module('denselayer{}'.format(i + 1), layer)
        
class _Transition(nn.Sequential):

    def __init__(self, num_caps, in_vect, out_vect):
        super().__init__()
        self.skip = convolutionalCapsule(in_capsules=num_caps, out_capsules=num_caps,
                                     in_channels=in_vect, out_channels=out_vect,
                                  stride=1, kernel=1, padding=0, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True)
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out1 = self.skip(x)
        out = out1.view(out1.shape[0] * out1.shape[1], out1.shape[2], out1.shape[3], out1.shape[4])
        out = self.avgpool(out)
        out = out.view(out1.shape[0], out1.shape[1], out1.shape[2], int(out1.shape[3]/2), int(out1.shape[4]/2))
        return out


class DenseNetModel(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=11,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1,
                 actvec_size=8):

        super().__init__()
        
        self.num_init_features = num_init_features
        self.actvec_size = actvec_size
        self.num_caps = int(num_init_features/actvec_size)

        # First convolution
        self.input_features = [('conv1',
                          nn.Conv2d(n_input_channels,
                                    num_init_features,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False)),
                         ('norm1', nn.BatchNorm2d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.input_features.append(
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
        self.input_features = nn.Sequential(OrderedDict(self.input_features))
        self.features = nn.Sequential()
        # Each denseblock
        num_actvec = actvec_size
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_caps=self.num_caps,
                                num_input_features=num_actvec,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                actvec_size=actvec_size)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_actvec = num_actvec + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_caps = self.num_caps, 
                                    in_vect=num_actvec, 
                                    out_vect=num_actvec // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_actvec = num_actvec // 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(26, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )

        # Linear layer
        self.classifier = nn.Linear((self.num_caps * num_actvec) + 64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, metadata):
        input_features = self.input_features(x)
        input_features = input_features.view(input_features.shape[0], int(self.num_init_features/self.actvec_size), self.actvec_size, input_features.shape[-2], input_features.shape[-1])
        out = self.features(input_features)
        y = self.metadata_network(metadata)
        out = self.classifier(torch.cat((out.view(out.size(0), -1), y), dim=1))
        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNetModel(num_init_features=64,
                         growth_rate=8,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNetModel(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNetModel(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNetModel(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model
    
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.SubModel = generate_model(121)
        
#     seq_len, batch, input_size
    def forward(self, x, y):
        out = self.SubModel(x, y)
        return out.flatten()    
    
    
class TrainDenseAttnCap():

    def __init__(self, epochs=300, batch_size=32, torch_type=torch.float32, split=1, trial=1):
        super(TrainDenseAttnCap, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        self.split = split
        self.trial = str(trial)
        self.model_name = "DenseAttnCap"
        
        self.mse = torch.nn.MSELoss()
        self.model = Model().to(self.device, dtype=torch.float32)
        
        self.train_dataset = EvapoDataset(split=self.split, train=True)
        self.test_dataset = EvapoDataset(split=self.split, train=False)
        
        self.dataset_size = len(self.train_dataset)
        self.indices = list(range(self.dataset_size))
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=5)
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=32, num_workers=5)
        self.final_accuracy_loader = torch.utils.data.DataLoader(self.test_dataset)
        
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.9)
        self.ssebop_model = torch.load("MLBOP_" + str(self.split) + ".pt")
    
    def train(self):
        timers = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for ind, (img_seq, true_et, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, date, lon, lat, file_name) in enumerate(self.train_loader):
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                true_et = true_et.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                clim = clim.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                lat = lat.to(device=self.device, dtype=torch.float32)
                lon = lon.to(device=self.device, dtype=torch.float32)
                ssebop_ET, _ = self.ssebop_model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                ssebop_ET = ssebop_ET.reshape(ssebop_ET.shape[0], -1)
                output = self.model(img_seq[:, 0:11], torch.cat((ssebop_ET, clim, veg, day_sin, day_cos, x_coord, y_coord, z_coord, elev), dim=1))
                loss = self.mse(output, true_et)
                loss.backward()
                self.opt.step()
#                 print("===> " + str(ind + 1) + "/" + str(int(self.dataset_size/self.batch_size)) + ", " + str(loss))
            self.sched.step()
            timers.append((time.time() - start_time))
            print("--- %s seconds ---" % (sum(timers)/len(timers)))
            print("Epoch " + str(epoch + 1))
#             self.test_accuracy_et = self.test(epoch)
#             print("Epoch " + str(epoch + 1) + ", Test " + self.test_accuracy_et)
        self.final_accuracy(epoch)
        torch.save(self.model, self.model_name + "_" + self.trial + "_" + str(self.split) +  ".pt" )
        
    def test(self, epoch):
        with torch.no_grad():
            correct = 0
            counter = 0
            for img_seq, true_et, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, date, lon, lat, file_name in self.test_loader:
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                true_et = true_et.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                clim = clim.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                lat = lat.to(device=self.device, dtype=torch.float32)
                lon = lon.to(device=self.device, dtype=torch.float32)
                ssebop_ET, _ = self.ssebop_model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                
                ssebop_ET = ssebop_ET.reshape(ssebop_ET.shape[0], -1)
                output = self.model(img_seq[:, 0:11], torch.cat((ssebop_ET, clim, veg, day_sin, day_cos, x_coord, y_coord, z_coord, elev), dim=1))
                correct += (torch.sum(torch.abs((output-true_et))))
                counter += output.shape[0]
            return str(round(float(correct.sum() / counter), 4))
        
        
        
    def final_accuracy(self, epoch):
        with open( self.model_name + "_"  + self.trial + "_" + str(self.split) + '.txt', 'w') as f:
            with torch.no_grad():
                et_correct = 0
                cfactor_correct = 0
                counter = 0
                for img_seq, true_et, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, date, lon, lat, file_name in self.final_accuracy_loader:
                    img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                    true_et = true_et.to(device=self.device, dtype=torch.float32)
                    veg = veg.to(device=self.device, dtype=torch.float32)
                    clim = clim.to(device=self.device, dtype=torch.float32)
                    day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                    day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                    x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                    y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                    z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                    elev = elev.to(device=self.device, dtype=torch.float32)
                    lat = lat.to(device=self.device, dtype=torch.float32)
                    lon = lon.to(device=self.device, dtype=torch.float32)
                    
                    
                    img_seq = torch.cat((img_seq, img_seq), 0)
                    true_et = torch.cat((true_et, true_et), 0)
                    veg = torch.cat((veg, veg), 0)
                    clim = torch.cat((clim, clim), 0)
                    day_sin = torch.cat((day_sin, day_sin), 0)
                    day_cos = torch.cat((day_cos, day_cos), 0)
                    x_coord = torch.cat((x_coord, x_coord), 0)
                    y_coord = torch.cat((y_coord, y_coord), 0)
                    z_coord = torch.cat((z_coord, z_coord), 0)
                    elev = torch.cat((elev, elev), 0)
                    date = (date[0], date[0])
                    lat = torch.cat((lat, lat), 0)
                    lon = torch.cat((lon, lon), 0)
                    
                    
                    
                    ssebop_ET, _ = self.ssebop_model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                    ssebop_ET = ssebop_ET.reshape(ssebop_ET.shape[0], -1)
                    output = self.model(img_seq[:, 0:11], torch.cat((ssebop_ET, clim, veg, day_sin, day_cos, x_coord, y_coord, z_coord, elev), dim=1))
                    
                    et_correct += (torch.sum(torch.abs((output-true_et))))/2
                    counter += output.shape[0]/2
                    
                    
                    f.write(file_name[0].split("/")[-1] + ', ' + str(float(output[0])) + ', ' + str(float(true_et[0])) + ', ' + str(float(torch.abs(output[0]-true_et[0]))) + "\n")
            f.close()

            
if __name__ == '__main__':
    
    trainer = TrainDenseAttnCap(split=int(1), trial=int(1))
    trainer.train()
#     start_range = 0
#     stop_range = 86
#     redoDense = [['1', '5'],
#                  ['26', '4'],
#                  ['27', '4'],
#                  ['27', '5'],
#                  ['58', '3'],
#                  ['6', '2'],
#                  ['60', '1'],
#                  ['63', '2'],
#                  ['65', '5'],
#                  ['66', '3'],
#                  ['73', '5'],
#                  ['76', '3'],
#                  ['8', '4'],
#                  ['80', '1'],
#                  ['80', '4'],
#                  ['86', '1'],
#                  ['90', '1'],
#                  ['92', '2']]
#     if (socket.gethostname() == "lattice-211"):
#         start_range = 0
#         stop_range = 2 
#     elif (socket.gethostname() == "lattice-212"):
#         start_range = 2
#         stop_range = 4
#     elif (socket.gethostname() == "lattice-213"):
#         start_range = 4
#         stop_range = 6
#     elif (socket.gethostname() == "lattice-214"):
#         start_range = 6
#         stop_range = 7
#     elif (socket.gethostname() == "lattice-215"):
#         start_range = 7
#         stop_range = 8
#     elif (socket.gethostname() == "lattice-216"):
#         start_range = 8
#         stop_range = 9
#     elif (socket.gethostname() == "lattice-217"):
#         start_range = 9
#         stop_range = 10
#     elif (socket.gethostname() == "lattice-218"):
#         start_range = 10
#         stop_range = 11
#     elif (socket.gethostname() == "lattice-219"):
#         start_range = 11
#         stop_range = 12
#     elif (socket.gethostname() == "lattice-220"):
#         start_range = 12
#         stop_range = 13
#     elif (socket.gethostname() == "lattice-221"):
#         start_range = 13
#         stop_range = 14
#     elif (socket.gethostname() == "lattice-222"):
#         start_range = 14
#         stop_range = 15
#     elif (socket.gethostname() == "lattice-223"):
#         start_range = 15
#         stop_range = 16
#     elif (socket.gethostname() == "lattice-224"):
#         start_range = 16
#         stop_range = 17
#     elif (socket.gethostname() == "lattice-225"):
#         start_range = 17
#         stop_range = 18

#     if (socket.gethostname() == "lattice-196"):
#         start_range = 0
#         stop_range = 2 
#     elif (socket.gethostname() == "lattice-197"):
#         start_range = 2
#         stop_range = 4
#     elif (socket.gethostname() == "lattice-198"):
#         start_range = 4
#         stop_range = 6
#     elif (socket.gethostname() == "lattice-199"):
#         start_range = 6
#         stop_range = 7
#     elif (socket.gethostname() == "lattice-200"):
#         start_range = 7
#         stop_range = 8
#     elif (socket.gethostname() == "lattice-201"):
#         start_range = 8
#         stop_range = 9
#     elif (socket.gethostname() == "lattice-202"):
#         start_range = 9
#         stop_range = 10
#     elif (socket.gethostname() == "lattice-203"):
#         start_range = 10
#         stop_range = 11
#     elif (socket.gethostname() == "lattice-204"):
#         start_range = 11
#         stop_range = 12
#     elif (socket.gethostname() == "lattice-205"):
#         start_range = 12
#         stop_range = 13
#     elif (socket.gethostname() == "lattice-206"):
#         start_range = 13
#         stop_range = 14
#     elif (socket.gethostname() == "lattice-207"):
#         start_range = 14
#         stop_range = 15
#     elif (socket.gethostname() == "lattice-208"):
#         start_range = 15
#         stop_range = 16
#     elif (socket.gethostname() == "lattice-209"):
#         start_range = 16
#         stop_range = 17
#     elif (socket.gethostname() == "lattice-210"):
#         start_range = 17
#         stop_range = 18
    
    
#     for t, r in redoDense[start_range:stop_range]:
#         trainer = TrainCNN(split=int(r), trial=int(t))
#         trainer.train()
#         print("DONE " + str(t) + " TRIAL, " + str(r) + " SPLIT")
            
            

# if __name__ == '__main__':
#     start_range = 1
#     stop_range = 101
#     if (socket.gethostname() == "lattice-211"):
#         start_range = 3
#         stop_range = 11   
#     elif (socket.gethostname() == "lattice-212"):
#         start_range = 11
#         stop_range = 21
#     elif (socket.gethostname() == "lattice-213"):
#         start_range = 21
#         stop_range = 31
#     elif (socket.gethostname() == "lattice-214"):
#         start_range = 31
#         stop_range = 41
#     elif (socket.gethostname() == "lattice-215"):
#         start_range = 41
#         stop_range = 51
#     elif (socket.gethostname() == "lattice-216"):
#         start_range = 51
#         stop_range = 61
#     elif (socket.gethostname() == "lattice-217"):
#         start_range = 61
#         stop_range = 71
#     elif (socket.gethostname() == "lattice-218"):
#         start_range = 71
#         stop_range = 81
#     elif (socket.gethostname() == "lattice-219"):
#         start_range = 81
#         stop_range = 91
#     elif (socket.gethostname() == "lattice-220"):
#         start_range = 91
#         stop_range = 101
    
    
#     for t in range(start_range, stop_range):
#         for r in range(1, 6):
#             trainer = TrainCNN(split=r, trial=t)
#             trainer.train()
#             print("DONE " + str(r) + " SPLIT")
#         print("DONE " + str(t) + " TRIAL")
