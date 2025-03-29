import torch
import torch.nn as nn
import torchvision.models as models

# Lớp để load trọng số cho mô hình phân loại
class ClassificationModelLoader:
    def __init__(self, model_path, num_classes, device):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        # Khởi tạo mô hình ResNet18 (không sử dụng trọng số pretrained)
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(device)
    
    def load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Đã load trọng số mô hình phân loại từ {self.model_path}")
        return self.model
