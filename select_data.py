import os
import shutil
import re

def generate_patterns(max_outer, max_inner):
    patterns = []
    for i in range(1, max_outer + 1):
        for j in range(1, max_inner + 1):
            pattern = f"{i}_{j}"
            patterns.append(pattern)
    return patterns

def img_num(source_dir,degree):
    count = 0
    for filename in os.listdir(source_dir):
        count+= 1
    if count == 1449:
        print(degree+'true')
    else:
        print(degree+'false')

def collect_images(source_dir, target_dir,degree = '1_1'):
    """
    从源目录查找所有匹配的图片，并将它们移动到目标目录。
    :param source_dir: 源文件夹路径，包含原始图片。
    :param target_dir: 目标文件夹路径，用于存放符合条件的图片。
    """
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 正则表达式匹配符合 "NYU_{index}_1_1.jpg"
    pattern = re.compile(rf'NYU2_(\d+)_{degree}\.jpg$')
    # 遍历源目录中的所有文件
    count = 0
    
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            
            count +=1
            index = match.group(1)  # 提取数字索引
            new_filename = f'NYU2_{index}.jpg'  # 构建新的文件名
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_filename)  # 使用新的文件名构建目标路径
            # 复制文件
            shutil.copy(source_path, target_path)
            print(f'File {filename} moved to {target_path}')
    if count==1449:
        print(degree,'is ok')     
    else:
        print(degree,'is not ok')
    

    
source_directory = './data/data'
target_directory = './data'
degree_list = generate_patterns(7, 3)

for item in degree_list:
    img_num(target_directory+f'/{item}',item)
    # collect_images(source_directory, target_directory+f'/{item}',item)
    
    
