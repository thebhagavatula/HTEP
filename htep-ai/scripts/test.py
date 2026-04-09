from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("tests/")
plt.imshow(img, cmap="gray")
plt.axis("off")
