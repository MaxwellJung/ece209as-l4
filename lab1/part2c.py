# Evaluate 1) evaluate 8-bit weights from 8-bit QAT and 2) 6-bit weights quantized from 8-bit weights
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

i8 = torch.iinfo(torch.int8)

class StraightThroughEstimator(torch.autograd.Function):
    weight_min_f32 = -128/128
    weight_max_f32 = 127/128
    i_bits = 8

    @staticmethod
    def forward(ctx, input: Tensor):
        # Implement your quantization function here
        # print(f'{torch.min(input)=} {torch.max(input)=}')
        # Map f32 to fake i8
        q = (input - StraightThroughEstimator.weight_min_f32)/(StraightThroughEstimator.weight_max_f32-StraightThroughEstimator.weight_min_f32)*(i8.max-i8.min) + i8.min
        q = torch.clamp(torch.round(q), min=i8.min, max=i8.max)

        # Select first i_bits bits
        q = torch.floor(q/(2**(8-StraightThroughEstimator.i_bits)))
        # Map to range [-1, 1)
        q = q/(2**(StraightThroughEstimator.i_bits-1))
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass the gradients are returned directly without modification.
        # This is the key step of STE
        return grad_output

# To config the STE
def config_ste(i_bits=8):
    StraightThroughEstimator.i_bits = i_bits
    # print(f'Quantize weight range [{StraightThroughEstimator.weight_min_f32}, {StraightThroughEstimator.weight_max_f32}] to {i_bits}-bit fixed-point')

# To apply the STE
def apply_ste(input: Tensor):
    return StraightThroughEstimator.apply(input)

# When you want to quantize the weights, call the apply_ste function
# You need to use this function within the forward pass of your model in a custom Conv2d class.

class Conv2d_custom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, apply_ste(self.weight), self.bias)

# To apply quantization and STE, you will need to define a new Conv2d class that will be used to replace the default Conv2d class in the ResNet model.
# You should follow a similar approach as Lab 1 Part 1.
Conv2dClass = Conv2d_custom # Default Conv2d class from pytorch. Replace this line!

# The code here is more complicated that it needs to be as it can be used to define multiple ResNet models with different configurations.
# You can ignore the specifics of the model code (contents of BasicBlock and ResNet classes), you only need to replace the Conv2dClass variable.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dClass(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = Conv2dClass(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dClass(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dClass(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, out.size(2))  # Global Average Pooling. Unique to ResNet8.
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet8 variant
def ResNet8():
    return ResNet(BasicBlock, [1, 1, 1])

net = ResNet8().to(device)

# Hyperparameters
num_epochs = 20
batch_size = 128 # you can lower this to 64 or 32 to help speed up training.
learning_rate = 0.001
# how much to weight 6 bit loss over 8 bit
# loss_ratio = 1 indicates equal weight
loss_ratio = 9

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model, weights, loss function and optimizer
model = ResNet8().to(device)
model.load_state_dict(torch.load('./lab1/part2c.pth', map_location=device))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #0.001 is the default LR for Adam.

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    correct_8_bit = 0
    correct_6_bit = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        config_ste(i_bits=8)
        outputs_8_bit = model(inputs)
        config_ste(i_bits=6)
        outputs_6_bit = model(inputs)
        loss = (criterion(outputs_8_bit, targets) + loss_ratio*criterion(outputs_6_bit, targets))/(1+loss_ratio)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted_8_bit = outputs_8_bit.max(1)
        _, predicted_6_bit = outputs_6_bit.max(1)
        total += targets.size(0)
        correct_8_bit += predicted_8_bit.eq(targets).sum().item()
        correct_6_bit += predicted_6_bit.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Loss: {train_loss/(batch_idx+1):.3f} '
                  f'Train Acc: {100.*correct_8_bit/total:.3f}% (8-bit), {100.*correct_6_bit/total:.3f}% (6-bit)')
# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct_8_bit = 0
    correct_6_bit = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            config_ste(i_bits=8)
            outputs_8_bit = model(inputs)
            config_ste(i_bits=6)
            outputs_6_bit = model(inputs)
            loss = (criterion(outputs_8_bit, targets) + loss_ratio*criterion(outputs_6_bit, targets))/(1+loss_ratio)
            
            test_loss += loss.item()
            _, predicted_8_bit = outputs_8_bit.max(1)
            _, predicted_6_bit = outputs_6_bit.max(1)
            total += targets.size(0)
            correct_8_bit += predicted_8_bit.eq(targets).sum().item()
            correct_6_bit += predicted_6_bit.eq(targets).sum().item()
            
    # Print summary on a new line after progress bar
    print(f'Epoch {epoch+1}: '
          f'Loss: {test_loss/(batch_idx+1):.3f} '
          f'Test Acc: {100.*correct_8_bit/total:.3f}% (8-bit), {100.*correct_6_bit/total:.3f}% (6-bit)')
    return correct_8_bit/total, correct_6_bit/total

# Train the model
best_acc_8_bit = 0
best_acc_6_bit = 0
for epoch in range(num_epochs):
    train(epoch)
    acc_8_bit, acc_6_bit = test(epoch)
    
    # Save model if better than previous best
    average_acc = (acc_8_bit+acc_6_bit)/2
    best_average_acc = (best_acc_8_bit+best_acc_6_bit)/2
    if (average_acc > best_average_acc):
        print(f'Saving model, {average_acc=:.3f} > {best_average_acc=:.3f},  {100.*acc_8_bit:.3f}% (8-bit), {100.*acc_6_bit:.3f}% (6-bit)')
        best_acc_8_bit = acc_8_bit
        best_acc_6_bit = acc_6_bit
        torch.save(model.state_dict(), './lab1/part2c.pth')

print(f'Best test accuracy: {best_acc_8_bit*100:.2f}% (8-bit), {best_acc_6_bit*100:.2f}% (6-bit)')
print('Training completed! Model saved as lab1/part2c.pth')
