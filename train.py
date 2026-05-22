import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image




def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content_dir', type=str, default='sample-dataset/content-data', help='Location of content dataset')
    
    parser.add_argument('--style_dir', type=str, default='sample-dataset/style-data', help='Location of style dataset')
    
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth', help='Location of pre-trained VGG model')
    
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of the experiment')
    
    parser.add_argument('--final_size', type=int, default=256, help='Size of final image')
    
    parser.add_argument('--content_size', type=int, default=512, help='Size of content image')
    
    parser.add_argument('--style_size', type=int, default=512, help='Size of style image')
    
    parser.add_argument('--crop', action='store_true', default=False, help='Crop images')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay for training')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    
    parser.add_argument('--content_weight', type=float, default=1.0, help='Weight for content loss')
    
    parser.add_argument('--style_weight', type=float, default=10, help='Weight for style loss')
    
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging training status')
    
    parser.add_argument('--save_interval', type=int, default=2, help='Interval for saving model checkpoints')
    
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from checkpoint')
    
    parser.add_argument('--decoder_path', type=str, default=None, help='Location of decoder checkpoint for resuming training')
    
    parser.add_argument('--optimizer_path', type=str, default=None, help='Location of optimizer checkpoint for resuming training')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
            
    content_transform = get_transforms(args.content_size, args.crop, args.final_size)
    style_transform = get_transforms(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, transform=content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, transform=style_transform)
    
    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=torch.cuda.is_available(),
                                    drop_last=True)
    
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)
    
    print(f'Content images: {len(content_dataset)} | batches/epoch: {len(content_dataloader)}')
    print(f'Style images: {len(style_dataset)} | batches/epoch: {len(style_dataloader)}')
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch))
    
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
    
    print('Starting training...')
    
    mse_loss = torch.nn.MSELoss()
    
    encoder.eval()  # Set encoder to evaluation mode
    
    running_loss = None
    running_closs = None
    running_sloss = None
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))
        
        running_loss = 0
        running_closs = 0
        running_sloss = 0
        
        for content_batch, style_batch in progress_bar:
            
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
    
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            
            g = decoder(t)

            g_feats = encoder(g)
            
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            
            loss_s = 0
            for gf, sf in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(gf)
                s_mean, s_std = calc_mean_std(sf)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
                
            loss_s *= args.style_weight
            
            loss = loss_c + loss_s
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            
            progress_bar.set_description(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Content Loss: {loss_c.item():.4f}, Style Loss: {loss_s.item():.4f}') 
            
        scheduler.step()
        
        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        
        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(f'Epoch {epoch + 1}, Loss: {running_loss:.4f}, Content Loss: {running_closs:.4f}, Style Loss: {running_sloss:.4f}')
            
        if (epoch + 1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_epoch_{epoch + 1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_epoch_{epoch + 1}.pth')
            
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_epoch_{epoch + 1}.png', nrow=args.batch_size)
                
    


if __name__ == '__main__':
    main()
