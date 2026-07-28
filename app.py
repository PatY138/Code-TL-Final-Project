from flask import Flask, request, jsonify
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "avif", "webp", "svg"}

# Serve static files from the current project folder.
app = Flask(__name__, static_folder='.', static_url_path='')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#MY CODE:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #defined 10 convolution layers with 10 separate batchnorms inbetween
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256,  kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256,  kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512,  kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.bn7= nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        #used global average pool layer 
        self.globalaveragepool = nn.AdaptiveAvgPool2d((1,1))

        self.pool = nn.MaxPool2d(2, 2)
      
        self.fc1 = nn.Linear(512, 256)

        self.bn11 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.gelu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = F.gelu(self.bn7(self.conv7(x)))
        x = F.gelu(self.bn8(self.conv8(x)))
        x = F.gelu(self.bn9(self.conv9(x)))
        x = F.gelu(self.bn10(self.conv10(x)))
        x = self.globalaveragepool(x)
        x = torch.flatten(x, 1)
        x = F.gelu(self.bn11(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()
net.to(device)
weights = torch.load('uploads/cifar10_model_weights.pth', map_location=torch.device('cpu'))
net.load_state_dict(weights)

def inference(image_file):
  img = Image.open(image_file).convert('RGB')
  test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.RandAugment(num_ops=2, magnitude=9),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  probs = []
  net.eval()
  with torch.no_grad():
    for _ in range(10):
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        output = net(img_tensor)
        probs.append(F.softmax(output, dim=1))
  
  final_output = torch.stack(probs).mean(0)
  _, predicted = torch.max(final_output, 1)
  return classes[predicted[0].item()]

#MY CODE END
@app.route('/')
def home():
    return app.send_static_file('homepage.html')


@app.route('/index.html')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Upload a valid image file.'}), 400

    label = inference(file)
    return jsonify({'label': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
