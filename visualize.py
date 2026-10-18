import os
from pathlib import Path

from utils.img_read_save import restore_color


if __name__ == "__main__":
    model_name = "CDDFuse_Exp0"
    dataset_name = "RoadScene"
    # image_name = "FLIR_04602.png"
    src_dir = Path("test_result") / model_name / dataset_name
    for image_name in os.listdir(Path("test_result") / model_name / dataset_name):
        fused_gray_path = src_dir / image_name
        if dataset_name == "RoadScene":
            image_name = image_name.replace(".png", ".jpg")
        visible_path = Path("test_img") / dataset_name / "vi" / image_name
        output_path = Path("rgb") / model_name / dataset_name / image_name
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        restore_color(fused_gray_path=fused_gray_path, visible_path=visible_path, output_path=output_path)