from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.files:
            raise ValueError(f"No image files found in dataset root: {root}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    
def get_transforms(size, crop, final_size):
    transform_list = []

    # Ensure the shorter side is large enough before cropping to final_size.
    safe_short_side = final_size if not size or size <= 0 else max(size, final_size)
    transform_list.append(transforms.Resize(safe_short_side))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.CenterCrop(final_size))
        
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)   

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch_size, channels, height, width]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch_size, channels, height, width]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std