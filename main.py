import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 1. Swin Transformer Backbone (or ResNet-50 as an alternative)
# class Backbone(nn.Module):
#     def __init__(self, use_resnet=True):
#         super(Backbone, self).__init__()
#         if use_resnet:
#             self.base = models.resnet50(pretrained=True)
#             self.feature_dim = 2048
#         else:
#             from transformers import SwinTransformerModel
#             self.base = SwinTransformerModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
#             self.feature_dim = 768

#     def forward(self, x):
#         return self.base(x)

class Backbone(nn.Module):
    def __init__(self, use_resnet=True):
        super(Backbone, self).__init__()
        if use_resnet:
            resnet = models.resnet50(pretrained=True)
            self.base = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
            self.feature_dim = 2048
        else:
            from transformers import SwinTransformerModel
            self.base = SwinTransformerModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.feature_dim = 768

    def forward(self, x):
        features = self.base(x)  # Extract features
        if len(features.shape) == 4:  # For ResNet, we get (batch, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # Convert to (batch, 2048)
        return features

# 2. Temporal Attention Module
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, features):
        # features shape: (batch_size, time_steps, feature_dim)
        weights = F.softmax(self.attention(features), dim=1)  # Attention weights
        context = torch.sum(weights * features, dim=1)  # Weighted sum
        return context

# 3. Video ReID Model
class VideoReIDModel(nn.Module):
    def __init__(self, use_resnet=True):
        super(VideoReIDModel, self).__init__()
        self.backbone = Backbone(use_resnet)
        self.temporal_attention = TemporalAttention(self.backbone.feature_dim)
        self.classifier = nn.Linear(self.backbone.feature_dim, 751)  # Assuming 751 identities (Market-1501 dataset)

    def forward(self, x):
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)  # Merge batch and time dimensions

        features = self.backbone(x)  # Extract features
        features = features.view(batch_size, time_steps, -1)  # Restore time dimension

        aggregated_features = self.temporal_attention(features)  # Temporal attention pooling
        logits = self.classifier(aggregated_features)  # Identity prediction

        # print(f"Logits shape: {logits.shape}")  # Should be (batch_size, 751)
        # print(f"Sample logits: {logits[0][:5]}")  # First 5 logits for a sample
        # print(f"Labels shape: {labels.shape}")  # Should be (batch_size,)
        # print(f"Sample labels: {labels[:5]}")  # First 5 labels

        return aggregated_features, logits
    

# 4. Loss Functions
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

# 5. Model Initialization
model = VideoReIDModel(use_resnet=True)
criterion = ReIDLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# def calculate_accuracy(logits, labels):
#     _, predictions = torch.max(logits, dim=1)  # Get the predicted class
#     correct = (predictions == labels).sum().item()  # Count correct predictions
#     accuracy = correct / labels.size(0) * 100  # Percentage
#     return accuracy

        # Random labels for testing

# model.eval()  # Set model to evaluation mode
# # with torch.no_grad():
# aggregated_features, logits = model(video_input)  # Forward pass
# acc = calculate_accuracy(logits, labels)         # Evaluate accuracy
# print(aggregated_features.shape)
# print(f"Rank-1 Accuracy: {acc:.2f}%")
def calculate_accuracy(logits, labels):
    _, predictions = torch.max(logits, dim=1)  # Get predicted class (Rank-1)
    correct = (predictions == labels).sum().item()  # Compare with labels
    accuracy = correct / labels.size(0) * 100
    return accuracy

# Training Loop
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

        # Forward pass
        aggregated_features, logits = model(video_input)
        
        # Loss Calculation
        loss = criterion(logits, (aggregated_features, aggregated_features, aggregated_features), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Accuracy Calculation
        acc = calculate_accuracy(logits, labels)
        
        epoch_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {correct/total * 100:.2f}%")


# model.eval()
# with torch.no_grad():
#     for i in range(5):  # Run 5 batches for evaluation
#         video_input = torch.randn(32, 8, 3, 224, 224)
#         labels = torch.randint(0, 751, (32,))
#         aggregated_features, logits = model(video_input)
#         acc = calculate_accuracy(logits, labels)
#         print(f"Batch {i+1} Accuracy: {acc:.2f}%")

# video_input = torch.randn(32, 8, 3, 224, 224)  # (batch_size, time_steps, channels, height, width)
# labels = torch.randint(0, 751, (32,))  
# # Use this after model forward pass
# aggregated_features, logits = model(video_input)
# acc = calculate_accuracy(logits, labels)
# print(f"Rank-1 Accuracy: {acc:.2f}%")


# print(model)  # Model structure overview


