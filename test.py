import os
def delete_png_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Provide the path to the folder where you want to delete the .png files
folder_path = r"C:\Users\Lamboss\Downloads\\fruit"
delete_png_files(folder_path)
