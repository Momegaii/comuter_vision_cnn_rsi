from cnn import *


test_img_path = "/content/drive/MyDrive/test.JPG"  # Or any valid image path
test_img_path2 = "/content/drive/MyDrive/test2.JPG"  # Or any valid image path


from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# Load and transform the image
img = Image.open(test_img_path)
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Load and transform the image
img2 = Image.open(test_img_path2)
img_tensor2 = transform(img2).unsqueeze(0)  # Add batch dimension


model.eval()  # Set to evaluation mode

with torch.no_grad():
    output = model(img_tensor2)
    predicted_class = torch.argmax(output, dim=1).item()
    class_name = train_dataset.classes[predicted_class]
    print(f"🧠 Prediction: {class_name}")


import matplotlib.pyplot as plt

plt.imshow(img2, cmap='gray')
plt.title(f"Predicted: {class_name}")
plt.axis('off')
plt.show()

torch.save(model.state_dict(), "/content/drive/My Drive/chart_cnn.pth")
