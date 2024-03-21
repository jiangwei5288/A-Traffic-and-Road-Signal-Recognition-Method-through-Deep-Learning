import matplotlib.pyplot as plt
import os
import shutil
import random

data_root = "D:/Road_landmarks_train_original"
print(os.listdir(data_root))
data_list = os.listdir(data_root)
data_list = sorted((data_list))
sample_n_list = []
count = 0
id_number = 0
for i in range(len(data_list)):
    id_name = data_list[i][:3]
    if (id_name == '%03d' % id_number):
        count = count + 1
    else:

        sample_n_list.append(count)
        id_number += 1
        count = 1
print(count)
print(sample_n_list)

import os
import shutil


def split_images_by_category(input_folder, output_root, category_counts_list):
    # 获取文件列表
    file_list = os.listdir(input_folder)

    # 创建输出根文件夹
    os.makedirs(output_root, exist_ok=True)

    # 创建输出文件夹并生成 category_counts 字典
    category_counts = {}
    for t_id, count in enumerate(category_counts_list, start=0):
        category = '%03d' % t_id
        category_folder = os.path.join(output_root, f'Category_{category}')
        os.makedirs(category_folder, exist_ok=True)  # 确保目标文件夹已经存在
        category_counts[category] = count

    # 记录每种图片类型已处理的数量
    processed_counts = {category: 0 for category in category_counts}

    # 将文件移动到不同的文件夹中
    for file_name in file_list:
        # 假设文件名格式为 "%03d_rest_of_the_name.png"
        try:
            t_id = int(file_name[:3])  # 提取前三位作为 t_id
            category = '%03d' % t_id
            category_folder = os.path.join(output_root, f'Category_{category}')

            source_path = os.path.join(input_folder, file_name)
            destination_path = os.path.join(category_folder, file_name)

            # 确保目标文件夹已经存在
            os.makedirs(category_folder, exist_ok=True)

            shutil.move(source_path, destination_path)

            processed_counts[category] += 1

            # 检查每种图片类型是否已处理完毕
            if processed_counts[category] == category_counts[category]:
                print(f'类别 {category} 完成。')
        except ValueError:
            print(f"Ignoring invalid file name: {file_name}")


input_folder = 'D:/Road_landmarks_train_original'
output_root = 'D:/Road_landmarks_original_categorized'

split_images_by_category(input_folder, output_root, sample_n_list)


def split_files(input_folder, output_folder, split_ratio=0.2):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for category_folder in subfolders:
        category_path = os.path.join(input_folder, category_folder)
        output_category_path = os.path.join(output_folder, category_folder)

        # 创建输出子文件夹
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        # 计算要分配到测试集的文件数量
        num_test_files = max(1, int(len(files) * split_ratio))

        # 随机选择测试集文件
        test_files = random.sample(files, min(num_test_files, len(files)))

        for file in test_files:
            src_path = os.path.join(category_path, file)
            dest_path = os.path.join(output_category_path, file)
            shutil.move(src_path, dest_path)

        print(f"Category '{category_folder}': {num_test_files} files moved to test set.")


if __name__ == "__main__":
    input_folder = "D:/test1"  # 替换成你的输入文件夹路径
    output_folder = "D:/test2"  # 替换成你的输出文件夹路径
    split_ratio = 0.2  # 测试集比例

    split_files(input_folder, output_folder, split_ratio)
# 将每个文件夹 2:8 分，分别测试  记录文件名（train和validation）
# 训练后将图片裁剪成同样的大小，resize到200*200，同时保留文件名
