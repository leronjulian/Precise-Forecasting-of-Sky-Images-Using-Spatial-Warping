import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from LiteFlowNet import *
from losses import *
from skynet_Unet_model import *
from dataLoader import *
from torch.utils.data import DataLoader

'''
    ============ TO DO ==============
    * Add part to save model after each 10 epochs
    * Add arguments ? (GPU, hyperparameters, epochs, batchsize)
    (DONE)* Add files for liteflownet (correlation, saved model)
    * Change code to make it look like mine
'''

#HyperParameters##
input_channels = 12
output_channels = 3
alpha = 1
lam_int = 5.0
lam_gd = 0.00111
lam_op = 0.010
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
EPOCHS = 40
BATCH_SIZE = 8
LR = 0.0002
################

#Model Paths#
lite_flow_model_path='./network-sintel.pytorch'

INPUTS_PATH = "./xTrain_skip.h5"
TARGET_PATH = "./yTrain_skip.h5"

##############################################


#### Dataset Loader
trainLoader = DataLoader(DatasetFromFolder(INPUTS_PATH, TARGET_PATH), BATCH_SIZE = 8, shuffle=True)

# Trains model (on training data) and returns the training loss
def run_train(model, x, y, gd_loss, op_loss, int_loss, optimizer): # Add skip as a parameter here
    target = y
    model = model.train()

    G_output = model(x)
  
    # For Optical Flow
    inputs = x
    input_last = inputs[:, 9:,:,:].clone().cuda() #I_t

    
    pred_flow_esti_tensor = torch.cat([input_last, G_output],1) #(Predicted)
    gt_flow_esti_tensor = torch.cat([input_last, target],1) #(Ground Truth)
    

    flow_gt = batch_estimate(gt_flow_esti_tensor, flow_network)
    flow_pred = batch_estimate(pred_flow_esti_tensor, flow_network)

    
    g_op_loss = op_loss(flow_pred, flow_gt)
    g_int_loss = int_loss(G_output, target)
    g_gd_loss = gd_loss(G_output, target)

    g_loss = lam_gd*g_gd_loss + lam_op*g_op_loss + lam_int*g_int_loss


    optimizer.zero_grad()

    g_loss.backward()
    optimizer.step()
    
    return g_loss.item()


# Unet Generator
generator = SkyNet_UNet(input_channels, output_channels)


generator = torch.nn.DataParallel(generator, device_ids=[3, 1]) 
generator = generator.to(device)

# Optical Flow Network
flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval()


# Sochastic Gradient Descent Weight Updater
optimizer = optim.Adam(generator.parameters(), lr = LR)


# Training Loss
train_loss = []

# Validation Loss 
valid_loss = []

# Losses
gd_loss = Gradient_Loss(alpha, 3).to(device)
op_loss = Flow_Loss().to(device)
int_loss = Intensity_Loss(1).to(device)

# Training Part ...
num_images = 0
for epoch in tqdm(range(EPOCHS), position = 0, leave = True):
    print('Starting Epoch...', epoch + 1)
    
    trainLossCount = 0
    num_images = 0
    for i, data in enumerate(trainLoader):
        # Training
        inputs = Variable(data[0]).to(device) # The input data
        target = Variable(data[1]).float().to(device)
        
        num_images += inputs.size(0)
        
        # Trains model
        trainingLoss = run_train(generator, inputs, target, gd_loss, op_loss, int_loss, optimizer) # Add skip as a parameter here
        trainLossCount = trainLossCount + trainingLoss
        
    
    epoch_loss = trainLossCount/num_images
    train_loss.append(epoch_loss)

    print('Training Loss...')
    print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch + 1, i + 1, len(trainLoader), epoch_loss))

    if epoch % 10 == 0:
        PATH =  './Iteration' + str(epoch) + '.pt'
        torch.save(generator, PATH)
        print('Saved model iteration' +  str(epoch) + 'to -> ' + PATH)


        
     

        
print('Training Complete...')

PATH =  './finalModel.pt'
torch.save(generator, PATH)
print('Saved Finnal Model to -> ' + PATH)