from pathlib import Path


def check_file_path(path: str | Path, create: bool = False) -> bool:
    path = Path(path)

    if path.exists() and not path.is_file():
        raise ValueError(f"目标路径已被目录占用，无法作为文件使用：{path}")

    parent = path.parent

    if parent.exists():
        if not parent.is_dir():
            raise NotADirectoryError(f"路径的父级已存在，但它是一个文件而非目录：{parent}")
        return True
    else:
        if create:
            parent.mkdir(parents=True, exist_ok=True)
            return True
        else:
            raise FileNotFoundError(f"父目录不存在且未授权创建：{parent}")
