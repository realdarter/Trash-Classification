import os
base_dir = "/data/garbagedata"
images_dict = {}
image_extension = {'.jpg', '.jpeg', '.png'}

folder_name = os.path.basename(base_dir)
print(folder_name)

for root, dirs, files in os.walk(base_dir):
    folder_name = os.path.basename(root)


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
