import sys
import argparse
import pickle
from data_utils import DatasetLoader
import torch
import numpy as np
from model import SER_AlexNet, SER_AlexNet_Attention, SER_FCN_Attention
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
import torch.nn.functional as f
import os
import random
from collections import Counter
from torch.backends import cudnn
#from torch.utils.tensorboard import SummaryWriter


def main(args):
    
    # Aggregate parameters
    params={
            #model & features parameters
            'ser_model': args.ser_model,

            #training
            'val_id': args.val_id,
            'test_id': args.test_id,
            'num_epochs':args.num_epochs,
            'batch_size':args.batch_size,
            'lr':args.lr,
            'dropout':args.dropout,
            'random_seed':args.seed,
            'use_gpu':args.gpu,
            
            #best mode
            'save_label': args.save_label,
            
            #parameters for tuning
            'oversampling': args.oversampling,
            'fcsize': args.fcsize,
            'scaler': args.scaler,
            'shuffle': args.shuffle,
            'pretrained': args.pretrained,
            'augment' : args.augment,
            'mixup' : args.mixup,
            'find_lr': args.find_lr
            }

    print('*'*40)
    print(f"\nPARAMETERS:\n")
    print('*'*40)
    print('\n')
    for key in params:
        print(f'{key:>15}: {params[key]}')
    print('*'*40)
    print('\n')

    #set random seed
    seed_everything(params['random_seed'])

    # Load dataset
    with open(args.features_file, "rb") as fin:
        features_data = pickle.load(fin)

    dataloader = DatasetLoader(features_data,
                               val_speaker_id=args.val_id,
                               test_speaker_id=args.test_id,
                               scaling=args.scaler,
                               oversample=args.oversampling,
                               augment=args.augment
                               )

    # Train
    train_stat = train(dataloader, params, save_label=args.save_label)

    return train_stat


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a SER AlexNet model in an iterative-based manner with "
                    "pyTorch and IEMOCAP dataset.")

    #Features
    parser.add_argument('features_file', type=str,
        help='Features extracted from `extract_features.py`.')
    
    #Model
    parser.add_argument('--ser_model', type=str, default='fcn_attention',
        help='SER model to be loaded')
    
    #Training
    parser.add_argument('--val_id', type=str, default='1F',
        help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default='1M',
        help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=200,
        help='Number of training epochs.') 
    parser.add_argument('--batch_size', type=int, default=32,
        help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, 
        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2,
        help='Dropout probability')
    parser.add_argument('--seed', type=int, default=100,
        help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=1,
        help='If 1, use GPU')
    
    #Best Model
    parser.add_argument('--save_label', type=str, default=None,
        help='Label for the current run, used to save the best model '
             'and Tensorboard label.')

    #Parameters for model tuning
    parser.add_argument('--oversampling', action='store_true',
        help='By default, no oversampling is applied to training dataset.'
             'Set this to true to apply random oversampling to balance training dataset')
             
    parser.add_argument('--fcsize', type=int, default=256,
        help='Size of output linear layer')
    
    parser.add_argument('--scaler', type=str, default='standard',
        help='Scaler type for dataset normalization. Available are'
             ' <standard> and <minmax>')
    
    parser.add_argument('--shuffle', action='store_true',
        help='By default, training dataloader does not shuffle every epoch'
             'Set this to true to shuffle the training data every epoch.')

    parser.add_argument('--pretrained', action='store_true',
        help='By default, AlexNet or FCN_Attention model weights are'
             'initialized randomly. Set this flag to initalize with '
             'pre-trained weights.')
    
    parser.add_argument('--augment', action='store_true',
        help='Set this to true to perform data augmentation at dataloader')
    
    parser.add_argument('--mixup', action='store_true',
        help='Set this to true to perform mixup at dataloader')

    parser.add_argument('--find_lr', action='store_true',
        help='Use LRFinder tool to plot loss vs iteration to find optimal learning rate.'
             'Set this flag to perform LRFinder test.')

    return parser.parse_args(argv)



def test(model, criterion, test_dataset, batch_size, device,
         return_matrix=False):

    """Test an SER model.

    Parameters
    ----------
    model
        PyTorch model.
    loss_function
    test_dataset : `speech_utils.ACRNN.torch.data_utils.TestLoader` instance
        The test dataset.
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    test_loss
        Description of returned object.

    """
    total_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    for i, batch in enumerate(test_loader):
        test_data_batch, test_labels_batch = batch

        # Send to correct device
        test_data_batch = test_data_batch.to(device)
        test_labels_batch = test_labels_batch.to(device, dtype=torch.long)
        
        # Forward
        test_preds_batch = model(test_data_batch)
        test_preds_segs.append(f.log_softmax(test_preds_batch, dim=1).cpu())
        
        #test loss
        test_loss = criterion(test_preds_batch, test_labels_batch)
        total_loss += test_loss.item()

    # Average loss
    test_loss = total_loss / (i+1)

    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = test_dataset.get_preds(test_preds_segs)
    
    # Make sure everything works properly
    assert len(test_preds) == test_dataset.n_actual_samples
    test_wa = test_dataset.weighted_accuracy(test_preds)
    test_ua = test_dataset.unweighted_accuracy(test_preds)

    results = (test_loss, test_wa*100, test_ua*100)
    
    if return_matrix:
        test_conf = test_dataset.confusion_matrix_iemocap(test_preds)
        return results, test_conf
    else:
        return results
    


def train(dataloader, params, save_label='default'):

    #get dataset
    train_dataset = dataloader.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=params['shuffle'])

    val_dataset = dataloader.get_val_dataset()
    test_dataset = dataloader.get_test_dataset()
    

    #setup Tensorboard
    #writer = SummaryWriter('runs/'+save_label)
    
    #select device
    if params['use_gpu'] == 1:
        device = torch.device("cuda:0")
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # Construct model, optimizer and criterion
    pretrained  = params['pretrained']
    ser_model   = params['ser_model']
    num_classes = dataloader.num_classes
    num_in_ch   = dataloader.num_in_ch
    
    if ser_model == 'fcn_attention':
        model = SER_FCN_Attention(num_classes=num_classes,
                                  dropout=params['dropout'],
                                  in_ch=num_in_ch,
                                  fcsize=params['fcsize'],
                                  pretrained=pretrained).to(device)
    elif ser_model == 'alexnet_attention':
        model = SER_AlexNet_Attention(num_classes=num_classes,
                                  dropout=params['dropout'],
                                  in_ch=num_in_ch,
                                  fcsize=params['fcsize'],
                                  pretrained=pretrained).to(device)
    elif ser_model == 'alexnet':
        model = SER_AlexNet(num_classes=num_classes,
                            in_ch=num_in_ch,
                            fcsize=params['fcsize'],
                            pretrained=pretrained).to(device)
    else:
        raise ValueError('No model found!')
    
    #trainloader = torch.utils.data.DataLoader(train_dataset, 
    #                            batch_size=params['batch_size'], 
    #                            shuffle=params['shuffle'])
    #dataiter = iter(trainloader)
    #data, labels = dataiter.next()

    #writer.add_graph(model,data)
    #writer.close()
    print(model.eval())
    print('\n')

    #Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    #If --find_lr flag is set, perform LRFinder test to find optimum 
    # learning rate for this model and configuration
    if params['find_lr'] is True:
        lr_range_test(model, optimizer, criterion,
                    train_dataset, val_dataset,params, device)
        exit()
    

    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    acc_format2 = "{:.02f}"
    best_val_wa = 0
    best_val_ua = 0
    save_path = save_label + '.pth'

    all_train_loss =[]
    all_train_wa =[]
    all_train_ua=[]
    all_val_loss=[]
    all_val_wa=[]
    all_val_ua=[]
    mixup = params['mixup']
    
    for epoch in range(params['num_epochs']):
        
        #get current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Train one epoch
        total_loss = 0
        train_preds = []
        target=[]
        model.train()
        for i, batch in enumerate(train_loader):
            
            train_data_batch, train_labels_batch = batch

            # Clear gradients
            optimizer.zero_grad()
            
            # Send data to correct device
            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device,dtype=torch.long)
            
            
            if mixup == True:
                # Mixup
                inputs, targets_a, targets_b, lam = mixup_data(train_data_batch, 
                        train_labels_batch, 0.2, use_cuda=torch.cuda.is_available(),
                        concat_ori=False)
                
                # Forward pass
                preds = model(inputs)

                # Loss
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                train_loss = loss_func(criterion, preds)
                
            else:
                # Forward pass
                preds = model(train_data_batch)

                # Loss
                train_loss = criterion(preds, train_labels_batch)
            
            # Compute the loss, gradients, and update the parameters
            total_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            
            # Accumulate batch results
            #train_preds.append(torch.argmax(f.log_softmax(preds,dim=1), dim=1).detach().cpu().numpy())
            #target.append(train_labels_batch.cpu().numpy())

        # Evaluate training data
        train_loss = total_loss / (i+1)
        #train_preds = np.concatenate(train_preds)
        #target = np.concatenate(target)
        #train_wa = train_dataset.weighted_accuracy(train_preds, target) * 100
        #train_ua = train_dataset.unweighted_accuracy(train_preds, target) * 100

        #print(loss_format.format(train_loss))
        all_train_loss.append(loss_format.format(train_loss))
        #all_train_wa.append(acc_format2.format(train_wa))
        #all_train_ua.append(acc_format2.format(train_ua))

        #Validation
        with torch.no_grad():
            val_result = test(
                model, criterion, val_dataset, 
                batch_size=64, #params['batch_size'],
                device=device)
        
            val_loss = val_result[0]
            val_wa = val_result[1]
            val_ua = val_result[2]

            # Update best model based on validation UA
            if val_wa > best_val_wa:
            #if val_ua > best_val_ua:
                best_val_ua = val_ua
                best_val_wa = val_wa
                best_val_loss = val_loss
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        all_val_loss.append(loss_format.format(val_loss))
        all_val_wa.append(acc_format2.format(val_wa))
        all_val_ua.append(acc_format2.format(val_ua))
        """
        # Plot in tensorboard
        writer.add_scalars('Loss',
                            {'train':train_loss,
                             'val':val_loss},
                            epoch + 1)
        
        writer.add_scalars('Accuracy',
                            {'WA_train':train_wa,
                             'WA_val':val_wa,
                             'UA_train':train_ua,
                             'UA_val':val_ua},
                            epoch + 1)
        """
        #print(f"Epoch {epoch+1}  (lr = {current_lr})\
        #	Loss: {loss_format.format(train_loss)} - {loss_format.format(val_loss)} - WA: {acc_format.format(train_wa)} - {acc_format.format(val_wa)} <{acc_format.format(best_val_wa)}> - UA: {acc_format.format(train_ua)} - {acc_format.format(val_ua)} <{acc_format.format(best_val_ua)}>")
        print(f"Epoch {epoch+1}  (lr = {current_lr})\
        	Loss: {loss_format.format(train_loss)} - {loss_format.format(val_loss)} - WA: {acc_format.format(val_wa)} <{acc_format.format(best_val_wa)}> - UA: {acc_format.format(val_ua)} <{acc_format.format(best_val_ua)}>")

    # Test on best model
    with torch.no_grad():
        model.load_state_dict(torch.load(save_path))

        #run validation once more to make sure model saved and loaded correctly
        val_result = test(
                model, criterion, val_dataset, 
                batch_size=1, #params['batch_size'],
                device=device)
        
        val_loss = val_result[0]
        val_wa = val_result[1]
        val_ua = val_result[2]


        test_result, confusion_matrix = test(
            model, criterion, test_dataset, 
            batch_size=1, #params['batch_size'],
            device=device, return_matrix=True)

        print("*" * 40)
        print("RESULTS ON TEST SET:")
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: "
              "{:.2f}".format(test_result[0], test_result[1], test_result[2]))
        print("Confusion matrix:\n{}".format(confusion_matrix[1]))   
        print(f'\n\nRE_VALIDATION') 
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: "
              "{:.2f}".format(val_loss, val_wa, val_ua))
        

    return(all_train_loss, all_train_wa, all_train_ua,
            all_val_loss, all_val_wa, all_val_ua,
            loss_format.format(test_result[0]), 
            acc_format2.format(test_result[1]),
            acc_format2.format(test_result[2]),
            confusion_matrix[0])


def lr_range_test(model, optimizer, criterion, train_dataset, val_dataset, params, device):
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=params['shuffle'])
    
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=False)
    
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, val_loader=val_loader, start_lr = 0.00001, end_lr=1, num_iter=100)
    lr_finder.plot(log_lr=False)
    lr_finder.reset()


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark=True
    cudnn.deterministic = True


def mixup_data(x, y, alpha=1.0, use_cuda=True, concat_ori=False):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    if concat_ori == True:
        mixed_x = torch.cat([x, mixed_x])
        y_a = torch.cat([y, y_a])
        y_b = torch.cat([y, y_b])

    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
