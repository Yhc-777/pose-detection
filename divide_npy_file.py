import numpy as np
import os

# 加载数据
sequences = np.load('./data/augmentation/keypoints_sequences_vae.npy')
labels = np.load('./data/augmentation/labels_vae.npy')

# 创建保存目录
output_dir = 'output/separated_sequences'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个序列并保存
for i in range(len(sequences)):
    # 保存序列数据
    sequence_filename = os.path.join(output_dir, f'sequence_{i:03d}.npy')
    np.save(sequence_filename, sequences[i])
    
    # 保存对应的标签
    label_filename = os.path.join(output_dir, f'label_{i:03d}.npy')
    np.save(label_filename, labels[i])

print(f"已将{len(sequences)}个序列分别保存到{output_dir}目录下")

# 验证保存的文件
# 随机加载一个文件来检查
test_idx = 0
loaded_sequence = np.load(os.path.join(output_dir, f'sequence_{test_idx:03d}.npy'))
loaded_label = np.load(os.path.join(output_dir, f'label_{test_idx:03d}.npy'))

print(f"\n验证第{test_idx}个序列:")
print(f"序列形状: {loaded_sequence.shape}")
print(f"标签值: {loaded_label}")
