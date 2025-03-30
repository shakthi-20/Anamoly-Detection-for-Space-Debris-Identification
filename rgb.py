import cv2
import os
from glob import glob
import numpy as np

# Set the folder containing TIFF images
input_folder = r"D:\DL\jwstimg"  # Change this to your folder path
output_folder = os.path.join(input_folder, "RGB_Converted")

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Get all TIFF images
tiff_files = glob(os.path.join(input_folder, "*.tif"))

# Process each TIFF file
for tiff_file in tiff_files:
    # Read the grayscale TIFF image
    gray_image = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)

    # Normalize to 0-255
    norm_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert grayscale to a false RGB using color mapping (apply COLORMAP_JET for visualization)
    color_image = cv2.applyColorMap(norm_image.astype(np.uint8), cv2.COLORMAP_JET)

    # Save the RGB image
    output_file = os.path.join(output_folder, os.path.basename(tiff_file).replace(".tif", "_rgb.png"))
    cv2.imwrite(output_file, color_image)

    print(f"Converted: {tiff_file} → {output_file}")

print("✅ Conversion Completed!")

