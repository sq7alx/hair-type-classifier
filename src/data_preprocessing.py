from PIL import Image
import os

# INPUT FOLDER
input_dir = "dataset/raw"
# OUTPUT FOLDER
output_dir = "dataset/processed"

target_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    counter = 1

    for filename in os.listdir(class_input_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png")):
            input_path = os.path.join(class_input_path, filename)
            # new file name
            output_filename = f"{class_name}-{counter}.png"
            output_path = os.path.join(class_output_path, output_filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")            
                    img = img.resize(target_size)       # scaling
                    img.save(output_path, "PNG")
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

            counter += 1
