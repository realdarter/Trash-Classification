import os
base_dir = "/data/garbage_classification"
images_dict = {}
image_extension = {'.jpg', '.jpeg', '.png'}

folder_name = os.path.basename(base_dir)

for root, dirs, files in os.walk(base_dir):
    folder_name = os.path.basename(root)
    print(folder_name)


"""

try:
    with open('paths.txt', 'r') as file:
        file_lines = file.readlines()
except Exception as x:
    print(x)


#print(paths)

for index, line in enumerate(file_lines):
    key, value = line.split('=')
    if("path" == key):
        current_path = value
        print(current_path)

"""
