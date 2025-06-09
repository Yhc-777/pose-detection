import os
from pathlib import Path

def get_tree_structure(directory, prefix="", is_last=True, max_depth=None, current_depth=0):
    """
    递归生成目录树结构
    
    Args:
        directory: 目录路径
        prefix: 当前行的前缀
        is_last: 是否是同级最后一个文件/目录
        max_depth: 最大递归深度
        current_depth: 当前递归深度
    """
    if max_depth is not None and current_depth > max_depth:
        return []
    
    try:
        path = Path(directory)
        if not path.exists():
            return [f"{prefix}[目录不存在: {directory}]"]
        
        # 获取当前目录名
        dir_name = path.name if path.name else str(path)
        
        # 根据是否是最后一个文件决定使用的符号
        connector = "└── " if is_last else "├── "
        
        # 构建当前行
        current_line = f"{prefix}{connector}{dir_name}/"
        lines = [current_line]
        
        # 获取目录内容并排序（目录在前，文件在后）
        try:
            items = list(path.iterdir())
            # 过滤掉隐藏文件和一些不需要的文件
            items = [item for item in items if not item.name.startswith('.') and 
                    item.name not in ['__pycache__', '.git', '.DS_Store']]
            
            # 按类型和名称排序：目录在前，文件在后，都按字母顺序
            directories = sorted([item for item in items if item.is_dir()], key=lambda x: x.name.lower())
            files = sorted([item for item in items if item.is_file()], key=lambda x: x.name.lower())
            items = directories + files
            
        except PermissionError:
            lines.append(f"{prefix}    [权限不足，无法访问]")
            return lines
        
        if not items:
            return lines
        
        # 决定下一级的前缀
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # 定义媒体文件扩展名
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.tiff', '.ico'}
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp', '.ts', '.mts'}
        media_extensions = image_extensions | video_extensions
        
        # 分类文件
        directories_list = [item for item in items if item.is_dir()]
        regular_files = [item for item in items if item.is_file() and item.suffix.lower() not in media_extensions]
        media_files = [item for item in items if item.is_file() and item.suffix.lower() in media_extensions]
        
        # 合并所有项目，保持原有顺序逻辑
        all_items = directories_list + regular_files
        
        # 处理媒体文件：只显示第一个，如果有多个就用...表示
        if media_files:
            all_items.append(media_files[0])  # 添加第一个媒体文件
            if len(media_files) > 1:
                # 创建一个虚拟的"..."项目
                class DummyItem:
                    def __init__(self):
                        self.name = "..."
                        self.is_dir = lambda: False
                        self.is_file = lambda: True
                all_items.append(DummyItem())
        
        # 处理每个项目
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            
            if hasattr(item, 'is_dir') and item.is_dir():
                # 递归处理子目录
                sub_lines = get_tree_structure(
                    item, next_prefix, is_last_item, max_depth, current_depth + 1
                )
                lines.extend(sub_lines)
            else:
                # 处理文件
                connector = "└── " if is_last_item else "├── "
                file_name = item.name
                
                lines.append(f"{next_prefix}{connector}{file_name}")
        
        return lines
        
    except Exception as e:
        return [f"{prefix}[错误: {str(e)}]"]

def main():
    """主函数"""
    # 获取当前目录
    current_dir = Path.cwd()
    
    print(f"{current_dir.name}/")
    
    # 生成目录树
    tree_lines = get_tree_structure(current_dir, is_last=True, max_depth=10)
    
    # 输出结果（跳过第一行，因为已经在上面打印了根目录）
    for line in tree_lines[1:]:
        print(line)

if __name__ == "__main__":
    main()
