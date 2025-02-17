import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#ResNet-50
class Backbone(nn.Module):
    def __init__(self, use_resnet=True):
        super(Backbone, self).__init__()
        if use_resnet:
            resnet = models.resnet50(pretrained=True)
            self.base = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 2048
        else:
            from transformers import SwinTransformerModel
            self.base = SwinTransformerModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.feature_dim = 768

    def forward(self, x):
        features = self.base(x) 
        if len(features.shape) == 4:
            features = features.squeeze(-1).squeeze(-1)
        return features
        
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, features):
        weights = F.softmax(self.attention(features), dim=1)
        context = torch.sum(weights * features, dim=1)
        return context

class VideoReIDModel(nn.Module):
    def __init__(self, use_resnet=True):
        super(VideoReIDModel, self).__init__()
        self.backbone = Backbone(use_resnet)
        self.temporal_attention = TemporalAttention(self.backbone.feature_dim)
        self.classifier = nn.Linear(self.backbone.feature_dim, 751)

    def forward(self, x):
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)

        features = self.backbone(x)  # Extract features
        features = features.view(batch_size, time_steps, -1) 

        aggregated_features = self.temporal_attention(features)
        logits = self.classifier(aggregated_features) 

        return aggregated_features, logits
    
class ReIDLoss(nn.Module):
    def __init__(self):
        super(ReIDLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def triplet_loss(self, embeddings, labels, margin=0.3):
        anchor, positive, negative = embeddings
        d_pos = (anchor - positive).pow(2).sum(1)
        d_neg = (anchor - negative).pow(2).sum(1)
        return F.relu(d_pos - d_neg + margin).mean()

    def forward(self, logits, embeddings, labels):
        ce_loss = self.cross_entropy(logits, labels)
        triplet_loss = self.triplet_loss(embeddings, labels)
        return ce_loss + triplet_loss

model = VideoReIDModel(use_resnet=True)
criterion = ReIDLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def calculate_accuracy(logits, labels):
    _, predictions = torch.max(logits, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for i in range(100):
        video_input = torch.randn(32, 8, 3, 224, 224) 
        labels = torch.randint(0, 751, (32,)) 

        optimizer.zero_grad()
        aggregated_features, logits = model(video_input)
        loss = criterion(logits, (aggregated_features, aggregated_features, aggregated_features), labels)
        
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        
        epoch_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {correct/total * 100:.2f}%")




# print(model)  # Model structure overview


