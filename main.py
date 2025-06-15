import copy
import json
import os
import sys

import pandas as pd
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ot
import numpy as np


from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer

import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)                      # Set seed for Python random module
    np.random.seed(seed)                   # Set seed for NumPy
    torch.manual_seed(seed)                # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)           # Set seed for PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # Set seed for all GPUs (if using multi-GPU)
    # torch.backends.cudnn.deterministic = True  # Make operations deterministic
    # torch.backends.cudnn.benchmark = False     # Disable benchmark to enforce determinism

# Example: Set a fixed seed
set_seed(42)

if len(sys.argv) == 2:
    
    training_properties = {
        "learning_rate": 0.001, 
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 10,
        "exp": 1,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 250  # How many training samples?
    }
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks 
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 64,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size.
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or cno_lrelu_torch or lrelu or 
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    #   darcy               : Darcy Flow

    which_example = sys.argv[1]
    #which_example = "shear_layer"

    # Save the models here:
    folder = "TrainedModels/"+"CNO_"+which_example+"_1"
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
p=2

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "darcy":
    example = Darcy(model_architecture_, device, batch_size, training_samples)
else:
    raise ValueError()
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER
test_loader = example.test_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

class OT_loss(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        elif which == 'emd':
            self.fn = lambda m, n, M: ot.emd(m, n, M)
        elif which == 'sinkhorn':
            self.fn = lambda m, n, M : ot.sinkhorn(m, n, M, 20.0, numItermax=5000)
        elif which == 'sinkhorn_knopp_unbalanced':
            self.fn = lambda m, n, M : ot.unbalanced.sinkhorn_knopp_unbalanced(m, n, M, 1.0, 1.0)
        else:
            pass
        self.use_cuda=use_cuda

    def __call__(self, source, target, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        mu = torch.from_numpy(ot.unif(source.size()[0]))
        nu = torch.from_numpy(ot.unif(target.size()[0]))
        M = torch.cdist(source, target)**2
        # mu = mu.to(dtype=torch.float32)
        # nu = nu.to(dtype=torch.float32)
        # M = M.to(dtype=torch.float32)
        N = M.detach().cpu()
        pi = self.fn(mu, nu, N)
        if type(pi) is np.ndarray:
            pi = torch.tensor(pi)
        elif type(pi) is torch.Tensor:
            pi = pi.clone().detach()
        pi = pi.cuda() if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WBM_loss(nn.Module):
    def __init__(self, eps):
        super(WBM_loss, self).__init__()
        self.lambd = nn.Parameter( 10+torch.rand(1)) # 
        self.eps = eps

    def forward(self, input_, target, model, var1, var2):

        return 0.01 + l1_loss(model(var1), var2) - self.lambd*torch.norm( input_ - var1, p=2) - self.lambd*torch.norm( target - var2, p=2)


if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
p=2
loss_up = OT_loss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.05 * epochs)    # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


var_rate = 1e-5
loss_d = WBM_loss(0.1)
loss_d = loss_d.to(device)

import pickle

import os

print(os.getcwd())  # Prints the current working directory

# with open("TrainedModels/CNO_shear_layer_1/model.pkl", "rb") as f:
# model = torch.load("TrainedModels/CNO_shear_layer_1/model.pkl")

# Load the state dict into the model

# model.load_state_dict(state_dict)
# model.eval()

# for step, (input_batch, output_batch) in enumerate(test_loader):
    
#     input_batch = input_batch.to(device)
#     if torch.rand(1)>0.8:
#       input_batch += 3*torch.randn_like(input_batch)
#     output_batch = output_batch.to(device)
#     output_pred_batch = model(input_batch)
    
#     loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100

#     print("test LOSS f ", loss_f)

# print("input ", torch.mean(torch.pow(input_batch, 2)))
# print("torch.std ", torch.mean(torch.pow(output_batch, 2)))

if 1:
  print("TRAIN LOADER ", len(train_loader))
  for epoch in range(epochs):
      with tqdm(unit="batch", disable=False) as tepoch:
          
          model.train()
          tepoch.set_description(f"Epoch {epoch}")
          train_mse = 0.0
          running_relative_train_mse = 0.0
          for step, (input_batch, output_batch) in enumerate(train_loader):

              # if step == 0:
              #   input_var = input_batch.clone() #   torch.zeros_like(input_batch, requires_grad=True) #  
              #   input_var = input_var.to(device)
              #   input_var.requires_grad_()
              #   output_var = output_batch.clone() #    torch.zeros_like(output_batch, requires_grad=True) #   
              #   output_var = output_var.to(device)
              #   output_var.requires_grad_()

              # input_batch = input_batch.to(device)
              # # print("max ", output_batch.shape)
              # # output_batch += 3*torch.randn_like(output_batch)
              # output_batch = output_batch.to(device)

              # loss_f = loss_d(input_batch, output_batch, model, input_var, output_var)
              # # Step 1: Differentiate w.r.t. the input
              # grad_input = torch.autograd.grad(loss_f, input_var, retain_graph=True)[0]
              # grad_output = torch.autograd.grad(loss_f, output_var, retain_graph=True)[0]
              
              # torch.nn.utils.clip_grad_norm_(grad_input, max_norm=1)
              # torch.nn.utils.clip_grad_norm_(grad_output, max_norm=1)
              # with torch.no_grad():
              #     input_var += var_rate * grad_input
              #     output_var += var_rate * grad_output

              optimizer.zero_grad()
              input_batch = input_batch.to(device)
              # print("max ", output_batch.shape)
              # output_batch += 3*torch.randn_like(output_batch)
              output_batch = output_batch.to(device)

              output_pred_batch = model(input_batch)

              output_batch = torch.squeeze(output_batch).reshape(len(output_batch), -1)
              output_pred_batch = torch.squeeze(output_pred_batch).reshape(len(output_pred_batch), -1)
              # print("output_batch ", output_batch.shape)
              # print("output_pred_batch ", output_pred_batch.shape)
              # print("SHAPE ", output_pred_batch[0].unsqueeze(-1).shape)

              if which_example == "airfoil": #Mask the airfoil shape
                  output_pred_batch[input_batch==1] = 1
                  output_batch[input_batch==1] = 1

              # loss_f = loss_d(input_batch, output_batch, model, input_var, output_var)  # 
              loss_f = loss_up(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)
              # loss_f = 0
              # for sam in range(len(output_batch)):
              #   loss_f += loss_up(output_pred_batch[sam].unsqueeze(-1), output_batch[sam].unsqueeze(-1)) / loss(torch.zeros_like(output_batch).to(device), output_batch)
              # loss_f /= len(output_batch) 

              loss_f.backward()
              optimizer.step()

              # with torch.no_grad():
              #   for param in model.parameters():
              #       param.clamp_(min=0)

              train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
              tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

          writer.add_scalar("train_loss/train_loss", train_mse, epoch)
          
          with torch.no_grad():
              model.eval()
              test_relative_l2 = 0.0
              train_relative_l2 = 0.0
              
              for step, (input_batch, output_batch) in enumerate(val_loader):
                  
                  input_batch = input_batch.to(device)
                  output_batch = output_batch.to(device)
                  output_pred_batch = model(input_batch)
                  
                  if which_example == "airfoil": #Mask the airfoil shape
                      output_pred_batch[input_batch==1] = 1
                      output_batch[input_batch==1] = 1
                  
                  loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100

                  print("LOSS f ", loss_f)
                  test_relative_l2 += loss_f.item()
              test_relative_l2 /= len(val_loader)

              for step, (input_batch, output_batch) in enumerate(train_loader):
                  input_batch = input_batch.to(device)
                  output_batch = output_batch.to(device)
                  output_pred_batch = model(input_batch)
                      
                  if which_example == "airfoil": #Mask the airfoil shape
                      output_pred_batch[input_batch==1] = 1
                      output_batch[input_batch==1] = 1

                      loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                      train_relative_l2 += loss_f.item()
              train_relative_l2 /= len(train_loader)
              
              writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
              writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

              if test_relative_l2 < best_model_testing_error:
                  best_model_testing_error = test_relative_l2
                  best_model = copy.deepcopy(model)
                  torch.save(best_model, folder + "/model.pkl")
                  writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                  counter = 0
              else:
                  counter+=1

          tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
          tepoch.close()

          with open(folder + '/errors.txt', 'w') as file:
              file.write("Training Error: " + str(train_mse) + "\n")
              file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
              file.write("Current Epoch: " + str(epoch) + "\n")
              file.write("Params: " + str(n_params) + "\n")
          scheduler.step()

      if counter>patience or epoch==100:
          for step, (input_batch, output_batch) in enumerate(test_loader):
              
              input_batch = input_batch.to(device)
              output_batch = output_batch.to(device)
              output_pred_batch = model(input_batch)
              
              loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100

              print("test LOSS f ", loss_f)
              test_relative_l2 += loss_f.item()
          test_relative_l2 /= len(val_loader)
          print("input ", torch.mean(torch.pow(input_batch, 2)))
          print("torch.std ", torch.mean(torch.pow(output_batch, 2)))
          print("Early Stopping")
      if counter>patience: 
        break
