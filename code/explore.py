import os 
from PIL import Image
import matplotlib.pyplot as plt 

real_files = os.listdir("data/train/REAL")
fake_files = os.listdir("data/train/FAKE")

print("Number of REAL training images: ", len(real_files))
print("Number of FAKE training images: ", len(fake_files))


fig, axes = plt.subplots(2, 5)

for i in range(0,5):
    real_img = Image.open("data/train/REAL/" + real_files[i])
    axes[0][i].imshow(real_img)

    fake_img = Image.open("data/train/FAKE/" + fake_files[i])
    axes[1][i].imshow(fake_img)


axes[0][0].set_title("REAL")
axes[1][0].set_title("FAKE")
plt.show()