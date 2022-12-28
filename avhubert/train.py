import os
import copy
import json
import argparse
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from dataset import ClassificationDataset
from net import Net

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.manual_seed(42)

def create_dataloader(image_dir, phase, batch_size):
    data = ClassificationDataset(image_dir, phase)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=data.collate_fn)
    return dataloader


def validate(dataloader, model, criterion, device):
    running_loss = []
    running_corrects = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader):
            video = data['video'].to(device)
            audio = data['audio'].to(device)
            labels = data['label'].to(device)

            outputs = model(video, audio, data['length'])
            loss = criterion(outputs, labels)
            preds = outputs.sigmoid().round()

            running_loss.append(loss.item())
            total += video.size(0)
            running_corrects += torch.sum(preds == labels.data)   

    val_loss = np.mean(running_loss)
    val_acc = running_corrects / total
    return val_loss, val_acc


def train(
    save_path,
    dataloaders,
    model,
    device,
    criterion,
    args
): 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    writer = SummaryWriter(save_path / 'summaries')    
    running_loss = []
    running_corrects = 0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
        print('-' * 10)

        # Iterate over data.
        total = len(dataloaders['train'])
        for idx, data in tqdm(enumerate(dataloaders['train']), total=total):
            step = epoch * total + idx + 1
            video = data['video'].to(device)
            audio = data['audio'].to(device)
            labels = data['label'].to(device)

            outputs = model(video, audio, data['length'])
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.sigmoid().round()
            running_loss.append(loss.item())
            running_corrects += torch.sum(preds == labels.data)
            
            if step % args.log_step == 0:
                train_loss = np.mean(running_loss)
                train_acc = running_corrects / (args.log_step * args.batch_size)
                # val_loss, val_acc = validate(dataloaders['val'], model, criterion, device)
                # model.train()

                print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
                # print(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

                writer.add_scalar('train/loss', train_loss, step)
                writer.add_scalar('train/acc', train_acc, step)
                # writer.add_scalar('val/loss', val_loss, step)
                # writer.add_scalar('val/acc', val_acc, step)

                running_loss = []
                running_corrects = 0

                # if val_acc > best_acc:
                #     best_acc = val_acc
                #     filename = 'best.pt'
                #     if torch.cuda.device_count() > 1:
                #         torch.save(model.module.state_dict(), save_path / filename)
                #     else: 
                #         torch.save(model.state_dict(), save_path / filename)

        filename = f'epoch_{epoch+1}.pt'
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), save_path / filename)
        else: 
            torch.save(model.state_dict(), save_path / filename)

    filename = 'last.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), save_path / filename)
    else: 
        torch.save(model.state_dict(), save_path / filename)


def main():
    # do not modify the configuration from the command line, 
    # otherwise there would be errors importing 'hubert_pretraining' and 'hubert'
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help='Path to npz train data directory')
    parser.add_argument('--backbone-path', type=str, help='pretrained checkpoint path, needed for training')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--fix-encoder', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='output', help='Path to directory for saving results')
    parser.add_argument('--log-step', type=int, default=1000)
    args = parser.parse_args()

    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'config', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Net(args.backbone_path, args.fix_encoder)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=False)
    model.to(device)

    dataloaders = {
        'train': create_dataloader(Path(args.root_dir), 'all', args.batch_size),
        # 'val': create_dataloader(Path(args.root_dir), 'val', args.batch_size),
    }

    criterion = nn.BCEWithLogitsLoss()
    TRAIN = True
    if TRAIN:
        train(
            save_path=save_path, 
            dataloaders=dataloaders,
            model=model,
            device=device,
            criterion=criterion,
            args=args
        )
    else:
        val_loss, val_acc = validate(dataloaders['val'], model, criterion, device)
        print(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

if __name__ == '__main__':
    main()