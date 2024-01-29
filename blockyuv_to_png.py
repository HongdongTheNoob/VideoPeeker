import os
import re
import numpy as np
import cv2

# Define the input and output directories
input_dir = "./blocks-pred"
output_dir = "./pngs-pred"
output_dir_2 = "./pngs-diff"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_2, exist_ok=True)

# Define a regular expression pattern to extract the last two integers from the filename
filename_pattern = r'(\d+)x(\d+)\.yuv'

# video file
video_file = open("D:/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv", "rb")
base_position = -1
frame = np.zeros(1)

# Iterate through files in the input directory
for filename in os.listdir(input_dir):
    # extract integers
    integers = re.findall(r'\d+', filename)
    integers = [int(i) for i in integers]
    if integers[4] < 32 or integers[5] < 32:
        continue

    filepath = os.path.join(input_dir, filename)

    # Check if the file is a .yuv file
    if not filename.endswith(".yuv"):
        continue

    # Extract the last two integers (x and y) from the filename using regular expressions
    match = re.search(filename_pattern, filename)
    if match:
        x, y = map(int, match.groups())
    else:
        continue

    # Read the file as an array of 16-bit unsigned integers
    try:
        data = np.fromfile(filepath, dtype=np.uint16) #byteswap(True)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        continue

    # Reshape the array as x-by-y
    data = data.reshape(y, x)

    # Normalize the data to 8-bit for grayscale image
    # data = (data / np.max(data) * 255).astype(np.uint8)
    data = (data // 4).astype(np.uint8)

    # Save the array as a grayscale PNG image in the output directory
    output_filename = os.path.splitext(filename)[0] + ".png"
    output_filepath = os.path.join(output_dir, output_filename)
    try:
        cv2.imwrite(output_filepath, data)
        # print(f"Saved {output_filepath}")
    except Exception as e:
        print(f"Error saving {output_filepath}: {e}")

    if base_position != 832*480*3//2*(integers[1]-1):
      video_file.seek(832*480*3//2*(integers[1]-1))
      base_position = 832*480*3//2*(integers[1]-1)
      content = video_file.read(832*480)
      frame = np.frombuffer(content, dtype=np.uint8).reshape((480, 832))
    block = frame[np.ix_(range(integers[3], integers[3]+integers[5]), range(integers[2], integers[2]+integers[4]))]
    diff = (block.astype(np.int16) - data.astype(np.int16) + 128).astype(np.uint8)

    # Save the array as a grayscale PNG image in the output directory
    output_filename = os.path.splitext(filename)[0] + ".png"
    output_filepath = os.path.join(output_dir_2, output_filename)
    try:
        cv2.imwrite(output_filepath, diff)
    except Exception as e:
        print(f"Error saving {output_filepath}: {e}")

video_file.close()