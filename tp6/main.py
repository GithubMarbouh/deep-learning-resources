import torch
from torchvision import models, transforms
import requests
from PIL import Image
from io import BytesIO


# Charger la liste des noms de classes (dans le bon ordre)
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]


# Charger le modèle VGG16 pré-entraîné
model = models.vgg16(pretrained=True)
model.eval()



# URL of the image
url = "https://images.pexels.com/photos/2071881/pexels-photo-2071881.jpeg"

# Get the image data
response = requests.get(url)
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Open the image
img = Image.open(BytesIO(response.content))


# Appliquer les transformations et ajouter une dimension de lot
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)


# Passer en avant
out = model(batch_t)


# Obtenir la classe prédite
_, predicted_idx = torch.max(out, 1)
predicted_class = imagenet_classes[predicted_idx.item()]


# Imprimer la classe prédite
print(predicted_class)