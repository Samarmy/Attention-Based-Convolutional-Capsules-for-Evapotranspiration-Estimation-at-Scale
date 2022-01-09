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
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
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
        return self.avgpool(mask)
    
class EvapoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, split=1):
        
        
        self.file_names = []
        for i in range(1, 6):
            if (train and (i != split)):
                self.file_names = self.file_names + glob.glob("split_data/" + str(split) + "/*")
                
            if (train is False and (i == split)):
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
    
    
    
    
class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(11, 256, 5, 1, padding=2, bias=False)
        )
        
        self.conv_caps = nn.Sequential(
            convolutionalCapsule(in_capsules=32, out_capsules=16, in_channels=32, out_channels=32,
                                  stride=2, padding=2, kernel=5, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=32, out_channels=32,
                                  stride=2, padding=2, kernel=5, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=32, out_channels=32,
                                  stride=2, padding=2, kernel=5, num_routes=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True)
            
        )
        
        
        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(26, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )
        
        self.linear_output = nn.Linear(512 + 64, 1)
        
        
       
    def forward(self, x, metadata):
        x = self.conv_input(x)
        x = x.view(x.size(0), 32, 32, 16, 16)
        x = self.conv_caps(x)
        x = x.view(x.size(0), -1)
        y = self.metadata_network(metadata)
        x = self.linear_output(torch.cat((x, y), dim=1))
        
        return x

    
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.SubModel = SubModel()
#         from torchsummary import summary
#         summary(self.SubModel, [[11, 32, 32], [26]])
        
#     seq_len, batch, input_size
    def forward(self, x, y):
        out = self.SubModel(x, y)
        return out.flatten()
    
    
class TrainAttnConvCap():

    def __init__(self, epochs=300, batch_size=32, torch_type=torch.float32, split=1, trial=1):
        super(TrainAttnConvCap, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        self.split = split
        self.trial = str(trial)
        self.model_name = "AttnConvCaps"
        
        self.mse = torch.nn.MSELoss()
        self.model = Model().to(self.device, dtype=torch.float32)
        
        self.train_dataset = EvapoDataset(split=self.split, train=True)
        self.test_dataset = EvapoDataset(split=self.split, train=False)
        
        self.dataset_size = len(self.train_dataset)
        self.indices = list(range(self.dataset_size))
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=5)
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=32, num_workers=5)
        self.final_accuracy_loader = torch.utils.data.DataLoader(self.test_dataset)
        
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25, gamma=0.5)
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
#                 print("===> " + str(ind + 1) + "/" + str(int(self.dataset_size/self.batch_size)))
            self.sched.step()
            timers.append((time.time() - start_time))
            print("--- %s seconds ---" % (sum(timers)/len(timers)))
            print("Epoch " + str(epoch + 1))
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
                    ssebop_ET, _ = self.ssebop_model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                    ssebop_ET = ssebop_ET.reshape(ssebop_ET.shape[0], -1)
                    output = self.model(img_seq[:, 0:11], torch.cat((ssebop_ET, clim, veg, day_sin, day_cos, x_coord, y_coord, z_coord, elev), dim=1))
                    et_correct += (torch.sum(torch.abs((output-true_et))))
                    counter += output.shape[0]
                    
                    
                    f.write(file_name[0].split("/")[-1] + ', ' + str(float(output[0])) + ', ' + str(float(true_et[0])) + ', ' + str(float(torch.abs(output[0]-true_et[0]))) + "\n")
            f.close()

if __name__ == '__main__':
    trainer = TrainAttnConvCap(split=int(1), trial=int(1))
    trainer.train()
    
#     start_range = 1
#     stop_range = 101
#     if (socket.gethostname() == "lattice-176"):
#         trainer = TrainCNN(split=1, trial=39)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-177"):
#         trainer = TrainCNN(split=2, trial=39)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-178"):
#         trainer = TrainCNN(split=3, trial=39)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-179"):
#         trainer = TrainCNN(split=4, trial=39)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-180"):
#         trainer = TrainCNN(split=5, trial=39)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-181"):
#         trainer = TrainCNN(split=1, trial=40)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-182"):
#         trainer = TrainCNN(split=2, trial=40)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-183"):
#         trainer = TrainCNN(split=3, trial=40)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-184"):
#         trainer = TrainCNN(split=4, trial=40)
#         trainer.train()
#     elif (socket.gethostname() == "lattice-185"):
#         trainer = TrainCNN(split=5, trial=40)
#         trainer.train()         
            
# if __name__ == '__main__':
    
#     start_range = 1
#     stop_range = 101
#     if (socket.gethostname() == "lattice-176"):
#         start_range = 1
#         stop_range = 3   
#     elif (socket.gethostname() == "lattice-177"):
#         start_range = 3
#         stop_range = 5
#     elif (socket.gethostname() == "lattice-178"):
#         start_range = 5
#         stop_range = 7
#     elif (socket.gethostname() == "lattice-179"):
#         start_range = 7
#         stop_range = 9
#     elif (socket.gethostname() == "lattice-180"):
#         start_range = 9
#         stop_range = 11
#     elif (socket.gethostname() == "lattice-181"):
#         start_range = 11
#         stop_range = 13
#     elif (socket.gethostname() == "lattice-182"):
#         start_range = 13
#         stop_range = 15
#     elif (socket.gethostname() == "lattice-183"):
#         start_range = 15
#         stop_range = 17
#     elif (socket.gethostname() == "lattice-184"):
#         start_range = 17
#         stop_range = 19
#     elif (socket.gethostname() == "lattice-185"):
#         start_range = 19
#         stop_range = 21
#     elif (socket.gethostname() == "lattice-186"):
#         start_range = 21
#         stop_range = 23
#     elif (socket.gethostname() == "lattice-187"):
#         start_range = 23
#         stop_range = 25
#     elif (socket.gethostname() == "lattice-188"):
#         start_range = 25
#         stop_range = 27
#     elif (socket.gethostname() == "lattice-189"):
#         start_range = 27
#         stop_range = 29
#     elif (socket.gethostname() == "lattice-190"):
#         start_range = 29
#         stop_range = 31
#     elif (socket.gethostname() == "lattice-191"):
#         start_range = 31
#         stop_range = 33   
#     elif (socket.gethostname() == "lattice-192"):
#         start_range = 33
#         stop_range = 35
#     elif (socket.gethostname() == "lattice-193"):
#         start_range = 35
#         stop_range = 37
#     elif (socket.gethostname() == "lattice-194"):
#         start_range = 37
#         stop_range = 39
#     elif (socket.gethostname() == "lattice-195"):
#         start_range = 39
#         stop_range = 41
#     elif (socket.gethostname() == "lattice-196"):
#         start_range = 41
#         stop_range = 43
#     elif (socket.gethostname() == "lattice-197"):
#         start_range = 43
#         stop_range = 45
#     elif (socket.gethostname() == "lattice-198"):
#         start_range = 45
#         stop_range = 47
#     elif (socket.gethostname() == "lattice-199"):
#         start_range = 47
#         stop_range = 49
#     elif (socket.gethostname() == "lattice-200"):
#         start_range = 49
#         stop_range = 51
#     elif (socket.gethostname() == "lattice-201"):
#         start_range = 51
#         stop_range = 53
#     elif (socket.gethostname() == "lattice-202"):
#         start_range = 53
#         stop_range = 55
#     elif (socket.gethostname() == "lattice-203"):
#         start_range = 55
#         stop_range = 57
#     elif (socket.gethostname() == "lattice-204"):
#         start_range = 57
#         stop_range = 59
#     elif (socket.gethostname() == "lattice-205"):
#         start_range = 59
#         stop_range = 61
#     elif (socket.gethostname() == "lattice-206"):
#         start_range = 61
#         stop_range = 63   
#     elif (socket.gethostname() == "lattice-207"):
#         start_range = 63
#         stop_range = 65
#     elif (socket.gethostname() == "lattice-208"):
#         start_range = 65
#         stop_range = 67
#     elif (socket.gethostname() == "lattice-209"):
#         start_range = 67
#         stop_range = 69
#     elif (socket.gethostname() == "lattice-210"):
#         start_range = 69
#         stop_range = 71
#     elif (socket.gethostname() == "lattice-211"):
#         start_range = 71
#         stop_range = 73
#     elif (socket.gethostname() == "lattice-212"):
#         start_range = 73
#         stop_range = 75
#     elif (socket.gethostname() == "lattice-213"):
#         start_range = 75
#         stop_range = 77
#     elif (socket.gethostname() == "lattice-214"):
#         start_range = 77
#         stop_range = 79
#     elif (socket.gethostname() == "lattice-215"):
#         start_range = 79
#         stop_range = 81
#     elif (socket.gethostname() == "lattice-216"):
#         start_range = 81
#         stop_range = 83
#     elif (socket.gethostname() == "lattice-217"):
#         start_range = 83
#         stop_range = 85
#     elif (socket.gethostname() == "lattice-218"):
#         start_range = 85
#         stop_range = 87
#     elif (socket.gethostname() == "lattice-219"):
#         start_range = 87
#         stop_range = 89
#     elif (socket.gethostname() == "lattice-220"):
#         start_range = 89
#         stop_range = 91
#     elif (socket.gethostname() == "lattice-221"):
#         start_range = 91
#         stop_range = 93
#     elif (socket.gethostname() == "lattice-222"):
#         start_range = 93
#         stop_range = 95
#     elif (socket.gethostname() == "lattice-223"):
#         start_range = 95
#         stop_range = 97
#     elif (socket.gethostname() == "lattice-224"):
#         start_range = 97
#         stop_range = 99
#     elif (socket.gethostname() == "lattice-225"):
#         start_range = 99
#         stop_range = 101
        
#     for t in range(start_range, stop_range):
#         for r in range(1, 6):
#             trainer = TrainCNN(split=r, trial=t)
#             trainer.train()
#             print("DONE " + str(r) + " SPLIT")
#         print("DONE " + str(t) + " TRIAL")                

# if __name__ == '__main__':
#     trainer = TrainCNN(split=int(1), trial=int(1))
#     trainer.train()
#     start_range = 70
#     stop_range = 101
#     if (socket.gethostname() == "lattice-216"):
#         start_range = 70
#         stop_range = 76 
#     elif (socket.gethostname() == "lattice-217"):
#         start_range = 76
#         stop_range = 82
#     elif (socket.gethostname() == "lattice-218"):
#         start_range = 82
#         stop_range = 88
#     elif (socket.gethostname() == "lattice-219"):
#         start_range = 88
#         stop_range = 94
#     elif (socket.gethostname() == "lattice-220"):
#         start_range = 94
#         stop_range = 101
    
    
#     for t in range(start_range, stop_range):
#         for r in range(1, 6):
#             trainer = TrainCNN(split=r, trial=t)
#             trainer.train()
#             print("DONE " + str(r) + " SPLIT")
#         print("DONE " + str(t) + " TRIAL")
