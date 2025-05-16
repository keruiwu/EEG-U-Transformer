from model import SeizureTransformer
import numpy as np
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm
import os


def ParseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--window_size', type=int, default=15360,
                        help='input sequence length of the model')
    parser.add_argument('--num_channel', type=int, default=19, 
                        help='number of channels')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha value for training data')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta value for training data')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for binary classification')
    
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dim_feedforward')
    parser.add_argument('--num_layers', type=int, default=8, help='num_layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='train epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-5)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    args.task_name = 'classification'
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args


class TrainingDataset(nn.Module):
    def __init__(self, data, label):
        super(TrainingDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.label[idx])

def get_trainingloader(data, label, batch_size=128):
    dataset = TrainingDataset(data, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def main():
    args = ParseArgs()
    data = np.load(f'../data/dataset/tusz_train_data_{args.alpha}_{args.beta}_{args.window_size}.npy')
    label = np.load(f'../data/dataset/tusz_train_label_{args.alpha}_{args.beta}_{args.window_size}.npy')
    val_data = np.load(f'../data/dataset/val_data_{args.window_size}.npy')
    val_label = np.load(f'../data/dataset/val_label_{args.window_size}.npy')
    
    # data = np.load('../main/seizure_detection/data/data_15360.npy')
    # label = np.load('../main/seizure_detection/data/label_15360.npy', mmap_mode='r')
    # val_data = np.load('../main/seizure_detection/data/val_data_15360.npy', mmap_mode='r')
    # val_label = np.load('../main/seizure_detection/data/val_label_15360.npy', mmap_mode='r')
    
    print('data', data.shape)
    print('label', label.shape)
    print('val_data', val_data.shape)
    print('val_label', val_label.shape)

    dataloader = get_trainingloader(data, label, batch_size=256)
    val_dataloader = get_trainingloader(val_data, val_label, batch_size=64)

    model = SeizureTransformer(
        in_channels=args.num_channel,
        in_samples=args.window_size,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
    )
    
    model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(f'cuda:{args.device_ids[0]}')

    loss_fn = nn.functional.binary_cross_entropy
    val_loss_fn = nn.functional.binary_cross_entropy
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=11, gamma=0.5)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print('='* 10, 'epoch', epoch, '=' * 10)
        print('-'*5, 'train', '-' * 5)
        model.train()
        progress = tqdm(dataloader, total=len(dataloader))

        for X, y_detect in progress:
            # print('X', X.shape)
            # print('y_detect', y_detect.shape)
            X = X.to(f'cuda:{args.device_ids[0]}')
            y_detect = y_detect.to(f'cuda:{args.device_ids[0]}')
            
            output = model(X)
            # print('detect', output[0].shape)
            detect_loss = loss_fn(output, y_detect)
            # print('detect_loss:', detect_loss.detach().cpu().item())
            progress.set_description(f'detect_loss: {detect_loss.detach().item():.4f}')
            
            optimizer.zero_grad()
            detect_loss.backward()
            optimizer.step()
        # scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Learning rate after epoch {epoch}: {current_lr:.6f}")
        print('-'*5, 'eval', '-' * 5)
        model.eval()
        progress = tqdm(val_dataloader, total=len(val_dataloader))
        avg_loss = 0
        loss_cnt = 0
        for X, y_detect in progress:
            X = X.to(f'cuda:{args.device_ids[0]}')
            y_detect = y_detect.to(f'cuda:{args.device_ids[0]}')

            output = model(X)
            detect_loss = val_loss_fn(output, y_detect)

            detect_loss = detect_loss.detach().cpu().item()
            progress.set_description(f'detect_loss: {detect_loss:.4f}')
            
            loss_cnt += 1
            avg_loss += detect_loss

        # remove unnecessary variables from gpu
        del X, y_detect, output
        torch.cuda.empty_cache()
        
        avg_loss /= loss_cnt
        print('avg_loss:', avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print('store best model with loss: ', best_loss)
            # store
            model_path = f'./ckp/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # model_name = model_path + f'0331_seizuretransformer_fully_res_elu_seq15360_head4_f512_layer8_ff2048_{args.lr}_{args.weight_decay}_{args.epochs}.pth'
            model_name = model_path + f'beta{args.beta}_alpha{args.alpha}_window{args.window_size}_dim{args.dim_feedforward}_layer{args.num_layers}.pth'

            if os.path.exists(model_name):
                os.remove(model_name)
            torch.save(model.module.state_dict(), model_name)   

if __name__ == '__main__':
    main()