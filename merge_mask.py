from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

image_folder = "/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign"

mask_1_images = [f for f in os.listdir(image_folder) if f.endswith("mask_1.png")]

for mask_1_image in mask_1_images:

    mask_image = mask_1_image.replace("mask_1.png", "mask.png")
    mask_image_path = os.path.join(image_folder, mask_image)
        
    if os.path.exists(mask_image_path):

        mask_1 = Image.open(os.path.join(image_folder, mask_1_image))
        mask = Image.open(mask_image_path)

        if mask_1.size != mask.size:
            print(f"Dimension mismatch error: {mask_1_image} and {mask_image} have different sizes.")
            continue
        
        mask_1_array = np.array(mask_1)
        mask_array = np.array(mask)
        
        if (mask_1_array.ndim == 3 and mask_1_array.shape[2] > 1):
            mask_1 = mask_1.convert('L')
            mask_1_array = np.array(mask_1)

        if (mask_array.ndim == 3 and mask_array.shape[2] > 1):
            mask = mask.convert('L')
            mask_array = np.array(mask)

        merged_array = np.logical_or(mask_1_array, mask_array).astype(np.uint8)
        
        merged_image = Image.fromarray(merged_array * 255)
        
        plt.imshow(merged_image)
        plt.show()
        
        os.remove(os.path.join(image_folder, mask_1_image))
        os.remove(mask_image_path)
        
        merged_image.save(mask_image_path)

        print(f"Merged and saved {mask_image}")
    else:
        print(f"Corresponding image {mask_image} not found for {mask_1_image}")

print("Task completed for mask 1")



mask_2 = Image.open("/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign/benign (195)_mask_2.png")
mask = Image.open("/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign/benign (195)_mask.png")

mask_2_array = np.array(mask_2)
mask_array = np.array(mask)


merged_array = np.logical_or(mask_2_array, mask_array).astype(np.uint8)
        
merged_image = Image.fromarray(merged_array * 255)
        
plt.imshow(merged_image)
plt.show()
        
os.remove("/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign/benign (195)_mask_2.png")
os.remove("/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign/benign (195)_mask.png")

merged_image.save("/Users/andywang/Desktop/Dataset_BUSI_with_GT 2/benign/benign (195)_mask.png")

print("Task completed for mask 2")


training_folder = "/Users/andywang/Desktop/Dataset_BUSI_with_GT 2 2/training"

benign = "/Users/andywang/Desktop/Dataset_BUSI_with_GT 2 2/benign"

normal = "/Users/andywang/Desktop/Dataset_BUSI_with_GT 2 2/normal"

benign_files = os.listdir(benign)
normal_files = os.listdir(normal)

for file in benign_files:
    source_file = os.path.join(benign, file)
    destination_file = os.path.join(training_folder, file)
    
    shutil.move(source_file, destination_file)
    
for file in normal_files:
    source_file = os.path.join(normal, file)
    destination_file = os.path.join(training_folder, file)
    
    shutil.move(source_file, destination_file)

print("All files moved from source folder to destination folder.")


