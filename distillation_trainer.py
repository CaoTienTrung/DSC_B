import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import a teacher model (a model with a large number of parameters and high accuracy)
from TeacherModel import TeacherModel
# import a student model (a small model with fewer parameters)
from StudentModel import StudentModel


# Distiller Class
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, x):
        # Forward pass through both teacher and student
        teacher_preds = self.teacher(x)
        student_preds = self.student(x)
        return teacher_preds, student_preds

    def distillation_loss(self, student_preds, teacher_preds, temperature, alpha):
        # Softmax with temperature for both teacher and student
        soft_teacher_preds = F.softmax(teacher_preds / temperature, dim=1)
        soft_student_preds = F.softmax(student_preds / temperature, dim=1)
        
        # Kullback-Leibler divergence for distillation loss
        distillation_loss = F.kl_div(F.log_softmax(student_preds / temperature, dim=1), 
                                      soft_teacher_preds, reduction='batchmean') * (temperature ** 2)
        return distillation_loss

    def student_loss(self, student_preds, targets):
        return F.cross_entropy(student_preds, targets)

    def compute_loss(self, student_preds, teacher_preds, targets, temperature, alpha):
        student_loss_value = self.student_loss(student_preds, targets)
        distillation_loss_value = self.distillation_loss(student_preds, teacher_preds, temperature, alpha)
        return alpha * student_loss_value + (1 - alpha) * distillation_loss_value

# Initialize models
teacher = TeacherModel()
student = StudentModel()

# Initialize distiller
distiller = Distiller(student=student, teacher=teacher)

# Optimizer
optimizer = optim.Adam(student.parameters(), lr=0.001)

# Training Loop for Teacher
def train_teacher(teacher, train_loader, optimizer, epochs=5):
    teacher.train()
    for epoch in range(epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = teacher(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} - Teacher Loss: {loss.item()}")

# Load train dataset
train_loader = ...

# Train teacher
train_teacher(teacher, train_loader, optimizer)

# Training Loop for Student with Distillation
def train_student(distiller, train_loader, optimizer, epochs=3, temperature=3, alpha=0.1):
    distiller.train()
    for epoch in range(epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass through teacher and student
            teacher_preds, student_preds = distiller(data)
            
            # Compute distillation loss and student loss
            loss = distiller.compute_loss(student_preds, teacher_preds, targets, temperature, alpha)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} - Student Loss: {loss.item()}")

# Load train dataset
train_loader = ...

# Train student with distillation
train_student(distiller, train_loader, optimizer)

# Evaluation for Student
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

# Load test dataset
test_loader = ...

# Evaluate the student model
evaluate_model(student, test_loader)
