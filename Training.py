import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF

import tqdm
import sys
import os

from NMNIST import *
from SHD import *
from Classifier_LIF import *
from Classifier_ReLIF import *

NUM_EPOCHS = 100

BATCH_SIZE = 64
NUM_THREADS = 2 # set lower! It needs a lot of shared memory

n_hidden = 128

def main():
    
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    

    #
    # Device
    #
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #
    # Dataset
    #
    
    if dataset_name == "NMMNIST":

        train_ds = NMMNIST(split="Train")
        test_ds = NMMNIST(split="Test")

        input_size = 68*68
        n_classes = 10
        
            

    elif dataset_name == "SHD":

        train_ds = SHD(split="train")
        test_ds = SHD(split="test")

        input_size = 700
        n_classes = 20


    #
    # Data loaders
    #

    train_loader = DataLoader(train_ds, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_THREADS, 
                              shuffle=True, 
                              drop_last=True,
                              prefetch_factor=16,
                              persistent_workers=False
                        )
    
    test_loader = DataLoader(test_ds, 
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_THREADS, 
                             shuffle=True, 
                             drop_last=True,
                             prefetch_factor=16,
                             persistent_workers=False
                        )


    #
    # Init Model
    #

    if model_name == "LIF":

        model = Classifier_LIF(input_size=input_size, n_hidden=n_hidden, n_classes=n_classes)
         
        run_id = f"{dataset_name}_{model_name}"

    elif model_name == "ReLIF":
        freq_max = int(sys.argv[3])
        model = Classifier_ReLIF(input_size=input_size, n_hidden=n_hidden, freq_max=freq_max, n_classes=n_classes)
        
        
        run_id = f"{dataset_name}_{model_name}_{freq_max}"

    else:
        print(f"invalid model_name: {model_name}")

    model.to(device)
    
    print(run_id)

    #
    # Logging
    #

    file_path = f"./logs/{run_id}/"

    writer = SummaryWriter(file_path)

    #
    # Train loop
    #
    for epoch in range(NUM_EPOCHS):
        
        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_accuracy
        if epoch == 0:

            train_loss, train_accuracy = model.test(train_loader, device)

        else:

            model.train()

            for x, targets in tqdm.tqdm(train_loader, position=0, leave=True):

                # x: (bs, t, d) = (64, 20, 1156)
                x = x.permute(1,0,2)

                # x: (t, bs, d) = (20, 64, 1156)

                # Transfer data to GPU (if available)
                x, targets = x.to(device), targets.to(device)

                # Reset gradients
                model.optimizer.zero_grad()

                # Forward pass
                spk_rec = model(x)
                spk_sum = spk_rec.sum(dim=0)
           
                # Calc loss
                loss = model.cce_rate_loss(spk_sum, targets)

                # Backprob
                loss.backward()

                # Update parameters
                model.optimizer.step()

                #
                # Update metrics
                #

                # Loss
                model.loss_metric.update(loss)

                # Accuracy
                accuracy = SF.accuracy_rate(spk_rec, targets)
                model.accuracy_metric.update(accuracy)

       
            train_loss = model.loss_metric.compute()
            train_accuracy = model.accuracy_metric.compute()


        test_loss, test_accuracy = model.test(test_loader, device)

        #
        # Output
        #
        print(f"      train_loss: {train_loss}")
        print(f"       test_loss: {test_loss}")
        print(f"  train_accuracy: {train_accuracy}")
        print(f"   test_accuracy: {test_accuracy}")
 

        #
        # Logging
        #
        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Test" : test_loss },
                            epoch)
        
        writer.add_scalars("Accuracy",
                            { "Train" : train_accuracy, "Test" : test_accuracy },
                            epoch)
        
        
        writer.flush()

        os.makedirs(f"./saved_models/{run_id}", exist_ok=True)
        torch.save(model.state_dict(), f"./saved_models/{run_id}/{epoch}")

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
