import numpy as np

BODY_PARTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


sequences = np.load('output/data/keypoints_sequences.npy')

# 查看数据基本信息
print(f"总视频数量: {sequences.shape[0]}")
print(f"每个序列帧数: {sequences.shape[1]}")  
print(f"每帧特征数: {sequences.shape[2]}")
print(f"数据类型: {sequences.dtype}")
print(f"数据范围: [{sequences.min():.3f}, {sequences.max():.3f}]")

# 解析第一个视频序列
video_0 = sequences[0]  # 形状: (20, 34)
print(f"\n第一个视频序列形状: {video_0.shape}")

# 解析第一帧的所有关键点
frame_0 = video_0[0]  # 形状: (34,)
print(f"\n第一帧关键点坐标:")
for i, part_name in enumerate(BODY_PARTS_NAMES):
    x, y = frame_0[i*2], frame_0[i*2+1]
    print(f"{part_name:15}: ({x:.3f}, {y:.3f})")

labels = np.load('output/data/labels.npy')
print(f"标签形状: {labels.shape}")
print(f"数据类型: {labels.dtype}")
print(f"唯一值: {np.unique(labels)}")






# 添加这部分来查看第一个视频序列对应的标签
print(f"\n第一个视频序列对应的标签: {labels[0]}")

# 创建标签映射字典来显示标签含义
label_mapping = {0: 'stand', 1: 'walk', 2: 'fall'}
print(f"第一个视频序列的动作类型: {label_mapping[labels[0]]}")

# 可以进一步查看更多视频序列的标签
print(f"\n前10个视频序列的标签和动作类型:")
for i in range(min(10, len(labels))):
    print(f"视频 {i}: 标签={labels[i]}, 动作={label_mapping[labels[i]]}")