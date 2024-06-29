import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your dataset
dataset_path = r'E:\Deep Learning\TENSORFLOW\rice_image_detection\Rice_Image_Dataset'  # Use raw string to avoid escape sequences
output_path = r'E:\Deep Learning\TENSORFLOW\rice_image_detection\output_dataset'  # Path to save the split datasets

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The system cannot find the path specified: '{dataset_path}'")

# Create output directories if they don't exist
train_folder = os.path.join(output_path, 'train')
test_folder = os.path.join(output_path, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get class names from the dataset directory
class_names = os.listdir(dataset_path)

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # List all files in the class directory
    files = os.listdir(class_path)
    files = [os.path.join(class_path, f) for f in files if os.path.isfile(os.path.join(class_path, f))]

    # Split the files into training and testing sets
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Create class directories in train and test folders
    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # Move the files to the respective directories
    for file in train_files:
        shutil.copy(file, train_class_folder)

    for file in test_files:
        shutil.copy(file, test_class_folder)

print("Dataset split and saved in train and test folders.")
