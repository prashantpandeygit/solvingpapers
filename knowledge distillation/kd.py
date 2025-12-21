import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

TEMPERATURE = 7 # to soft softamax outputs
ALPHA = 0.3 # weight difference between hard and soft loss

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), #image mnsit input from 28x28 to 784
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10), #10 classes for mnist
        )

    def forward(self, x): # forward pass logits, not probabilities
        return self.net(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        # smaller capacity, 1024 in teacher to 256 in student
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float, 
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # student log probabilities(temperature scaled) and teacher probabilities(temperature scaled)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    hard_loss = F.cross_entropy(student_logits, target)
    # soft loss using KL divergence to measure difference between student and teacher probabilities
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    # temperature squared as the above done tempeature scaling shrinks the gradients
    soft_loss = soft_loss * temperature * temperature
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    return total_loss, hard_loss, soft_loss

# data
def get_dataloader(train: bool):
    # transform to tensor and normalize to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train)


def train():
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    teacher = Teacher().to(DEVICE)
    student = Student().to(DEVICE)

    # pretrainining teacher
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=LEARNING_RATE)
    teacher.train()
    # just 3 epochs
    for _ in range(3):
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            teacher_optimizer.zero_grad()
            loss = F.cross_entropy(teacher(x), y)
            loss.backward()
            teacher_optimizer.step()

    # freezing teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # training student
    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0

        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)
            # calculate distillation loss, here kd happens.
            loss, hard_loss, soft_loss = distillation_loss(
                student_logits,
                teacher_logits,
                y,
                TEMPERATURE,
                ALPHA,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(student, test_loader)

        print(
            f'epoch {epoch+1:02d} | '
            f'loss: {avg_loss:.4f} | '
            f'accuracy: {accuracy:.2f}%'
        )

# evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return (correct / total) * 100.0


if __name__ == "__main__":
    train()

# implementation of knowledge distillation for mnist classification, but without cnn, i used a simple MLP.