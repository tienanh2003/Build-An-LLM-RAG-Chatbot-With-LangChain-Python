import torch
import torch.nn as nn

# Định nghĩa lớp DoubleConv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

# Định nghĩa kiến trúc UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

# Lớp để load trọng số cho mô hình segmentation
class SegmentationModelLoader:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = UNet(n_channels=3, n_classes=1).to(device)
    
    def load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Đã load trọng số mô hình segmentation từ {self.model_path}")
        return self.model
