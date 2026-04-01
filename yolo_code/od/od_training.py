"""
使用動態增強 + 符號移除訓練 YOLO
包含超音波影像符號清理功能
"""
# ⚠️ 重要：必須在導入 YOLO 之前先設置動態增強
from dynamic_augmentation import setup_dynamic_augmentation
setup_dynamic_augmentation()  # 先 patch，再導入 YOLO

from ultralytics import YOLO
import torch
from pathlib import Path


def train_with_symbol_removal():
    data_root = "/root/Desktop/polyp"
    output_root = "/datadrive/polyp/yolo_output"
    # data_root = "/Users/tony.tu/Desktop/戴承智慧/polyp"

    # 配置參數（完全照原本的方式）
    data_yaml = f'{data_root}/yolo_data/od/data.yaml'
    epochs = 100
    batch_size = -1
    img_size = 640
    project = f'{output_root}/runs/detect'
    name = '2026_04_01(1)'
    # name = "test"

    classes = [0]  # 只訓練結節類別（0）
    # 檢查設備
    device = 'mps' if torch.backends.mps.is_available() else '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")

    # 載入模型
    model = YOLO('yolo26s.pt')

    print("\n" + "="*60)
    print("使用動態增強 + 符號移除訓練 YOLO")
    print("="*60)
    print(f"數據集: {data_yaml}")
    print(f"訓練輪數: {epochs}")
    print(f"增強方式: 動態隨機 + 符號移除")
    print("="*60 + "\n")

    # 訓練
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        device=device,

        # 訓練參數
        patience=20,
        save=True,
        save_period=10,
        cache="disk",
        plots=True,

        # 混合模式：自定義影像增強 + YOLO 幾何變換
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,

        # 優化器
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # 其他
        val=True,
        workers=8,
        classes=classes,
        verbose=True,
    )

    print("\n" + "="*60)
    print("訓練完成！")
    print("="*60)
    print(f"最佳模型: {project}/{name}/weights/best.pt")

    # 評估
    print("\n評估模型性能...")
    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model, results


if __name__ == '__main__':
    model, results = train_with_symbol_removal()
