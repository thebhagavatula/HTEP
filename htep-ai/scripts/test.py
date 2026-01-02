from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("data/icr_training/cursive/train/labour/*.png")
plt.imshow(img, cmap="gray")
plt.axis("off")
