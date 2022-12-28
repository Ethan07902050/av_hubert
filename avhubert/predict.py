from dataset import ClassificationDataset
from net import Net
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import argparse

torch.manual_seed(42)

def predict(dataloader, model, device):
    pred_dict = {}
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader):
            video = data['video'].to(device)
            audio = data['audio'].to(device)

            outputs = model(video, audio, data['length'])
            pred = outputs.sigmoid().round()
            
            pred_dict[data['fname'][0]] = int(pred)

    return pred_dict


def write_csv(pred, outpath):
    with open(outpath, 'w') as f:
        f.write('Id,Predicted\n')
        for key, value in pred.items():
            f.write(f'{key},{value}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/ssd/dlcv/test_data')
    parser.add_argument('--backbone-path', type=str, default='/ssd/tunhsiang/DLCV/final/base_vox_iter5.pt')
    parser.add_argument('--model-path', type=str, default='/ssd/tunhsiang/DLCV/final/log/all/epoch_3.pt')
    parser.add_argument('--output-path', type=str, default='output/test.csv')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Net(args.backbone_path)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.to(device)

    data = ClassificationDataset(args.root_dir, mode='test')
    dataloader = DataLoader(data, batch_size=1, num_workers=4, collate_fn=data.collate_fn) 
    
    pred = predict(dataloader, model, device)
    write_csv(pred, args.output_path)

if __name__ == '__main__':
    main()