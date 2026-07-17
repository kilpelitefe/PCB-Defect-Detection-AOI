import os
from PIL import Image
import hashlib

# DeepPCB dataset root — adjust to your local path if needed
root_folder = "PCBData"

def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

missing_pairs = []
same_input_target = []
size_mismatch = []
valid_pairs = []

for root, dirs, files in os.walk(root_folder):
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in image_files:
        if "_test" not in file:
            continue

        target_file = file.replace("_test", "_temp")

        input_path = os.path.join(root, file)
        target_path = os.path.join(root, target_file)

        if not os.path.exists(target_path):
            missing_pairs.append((input_path, target_path))
            continue

        if md5(input_path) == md5(target_path):
            same_input_target.append((input_path, target_path))

        input_img = Image.open(input_path)
        target_img = Image.open(target_path)

        if input_img.size != target_img.size:
            size_mismatch.append((input_path, target_path, input_img.size, target_img.size))
        else:
            valid_pairs.append((input_path, target_path))

print("Valid pairs:", len(valid_pairs))
print("Missing targets:", len(missing_pairs))
print("Identical input-target pairs:", len(same_input_target))
print("Size-mismatched pairs:", len(size_mismatch))

print("\nExamples of missing targets:")
for input_path, expected_target in missing_pairs[:10]:
    print("Input:", input_path)
    print("Expected target:", expected_target)
    print()

print("\nExamples of identical input-target pairs:")
for x, y in same_input_target[:10]:
    print(x)
    print(y)
    print()
