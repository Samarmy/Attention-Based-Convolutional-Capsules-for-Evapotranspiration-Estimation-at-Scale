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
import time
import socket

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
    
class SubModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.in_planes = [16, 32, 64, 128]

        self.convolutions = nn.Sequential(nn.Conv2d(11, self.in_planes[0], kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(self.in_planes[0]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[0], self.in_planes[1], kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(self.in_planes[1]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[1], self.in_planes[2], kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(self.in_planes[2]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[2], self.in_planes[3], kernel_size=5, stride=1, padding=2),
                                        nn.BatchNorm2d(self.in_planes[3]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2)
        
        )
        
        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(26, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )
        
        
        self.fc = nn.Linear(512 + 64, 1)


    def forward(self, x, metadata):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        
        y = self.metadata_network(metadata)
        out = self.fc(torch.cat((x, y), dim=1))

        return out
    
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.SubModel = SubModel()
        
#     seq_len, batch, input_size
    def forward(self, x, y):
        out = self.SubModel(x, y)
        return out.flatten()
    
    
class TrainCNN():

    def __init__(self, epochs=300, batch_size=32, torch_type=torch.float32, split=1, trial=1):
        super(TrainCNN, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        self.split = split
        self.trial = str(trial)
        self.model_name = "CNN"
        
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
#                 print("HERE")
#                 print(output)
#                 print(true_et)
#                 print(output-true_et)
#                 print(torch.abs((output-true_et)))
#                 print(torch.abs((output-true_et)).mean())
#                 print(torch.sum(torch.abs((output-true_et))))
#                 print(correct)
                correct += (torch.sum(torch.abs((output-true_et))))
                counter += output.shape[0]
            return str(round(float(correct / counter), 4))
        
        
        
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
    
    start_range = 1
    stop_range = 101
    if (socket.gethostname() == "lattice-186"):
        trainer = TrainCNN(split=1, trial=39)
        trainer.train()
    elif (socket.gethostname() == "lattice-187"):
        trainer = TrainCNN(split=2, trial=39)
        trainer.train()
    elif (socket.gethostname() == "lattice-188"):
        trainer = TrainCNN(split=3, trial=39)
        trainer.train()
    elif (socket.gethostname() == "lattice-189"):
        trainer = TrainCNN(split=4, trial=39)
        trainer.train()
    elif (socket.gethostname() == "lattice-190"):
        trainer = TrainCNN(split=5, trial=39)
        trainer.train()
    elif (socket.gethostname() == "lattice-191"):
        trainer = TrainCNN(split=1, trial=40)
        trainer.train()
    elif (socket.gethostname() == "lattice-192"):
        trainer = TrainCNN(split=2, trial=40)
        trainer.train()
    elif (socket.gethostname() == "lattice-193"):
        trainer = TrainCNN(split=3, trial=40)
        trainer.train()
    elif (socket.gethostname() == "lattice-194"):
        trainer = TrainCNN(split=4, trial=40)
        trainer.train()
    elif (socket.gethostname() == "lattice-195"):
        trainer = TrainCNN(split=5, trial=40)
        trainer.train()                     
            
if __name__ == '__main__':
    trainer = TrainCNN(split=int(1), trial=int(1))
    trainer.train()
    
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
#     for t in range(1, 101):
#         for r in range(1, 6):
#             trainer = TrainCNN(split=r, trial=t)
#             trainer.train()
#             print("DONE " + str(r) + " SPLIT")
#         print("DONE " + str(t) + " TRIAL")
