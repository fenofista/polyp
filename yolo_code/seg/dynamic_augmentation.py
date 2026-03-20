"""
動態增強：每個 epoch 隨機應用不同的 augmentation
只需要定義一個 function，其他照 YOLO 原本流程
"""
import cv2
import numpy as np
import random
import time
from pathlib import Path




# ============================================================================
# 👇 在這裡定義你的增強 function（可以有多個）
# ============================================================================

def apply_clahe(image, prob=0.5):
    """CLAHE 增強 (推薦用於超音波)"""
    if random.random() > prob:
        return image

    if len(image.shape) == 2:  # 灰階
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    else:  # 彩色
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def apply_denoise(image, prob=0.3):
    """去噪"""
    if random.random() > prob:
        return image

    if len(image.shape) == 2:
        image = cv2.fastNlMeansDenoising(image, h=10)
    else:
        image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)

    return image


def apply_sharpen(image, prob=0.3):
    """銳化"""
    if random.random() > prob:
        return image

    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / 1.0
    image = cv2.filter2D(image, -1, kernel)

    return image


def apply_contrast(image, prob=0.4):
    """對比度和亮度調整"""
    if random.random() > prob:
        return image

    alpha = random.uniform(0.8, 1.3)  # 對比度
    beta = random.randint(-20, 20)     # 亮度
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image


def apply_blur(image, prob=0.2):
    """模糊（模擬不同掃描條件）"""
    if random.random() > prob:
        return image

    ksize = random.choice([3, 5])
    image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image


def apply_noise(image, prob=0.2):
    """高斯噪聲"""
    if random.random() > prob:
        return image

    noise = np.random.randn(*image.shape) * random.uniform(5, 15)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image



# ============================================================================
# 👇 組合所有增強（按照你想要的順序和機率）
# ============================================================================

def apply_custom_augmentation(image, print_timing=False):
    """
    主要的增強函數 - 會隨機應用多種增強
    每次調用都可能產生不同結果

    Args:
        image: numpy array (H, W, C) 或 (H, W)
        print_timing: 是否打印每個augmentation的執行時間

    Returns:
        augmented_image: numpy array
    """
    timing_info = {}

    # 按順序應用增強（每個都有自己的機率）

    start_time = time.time()
    image = apply_clahe(image, prob=0.5)        # 50% 機率
    timing_info['clahe'] = time.time() - start_time

    start_time = time.time()
    image = apply_contrast(image, prob=0.4)     # 40% 機率
    timing_info['contrast'] = time.time() - start_time

    start_time = time.time()
    image = apply_sharpen(image, prob=0.3)      # 30% 機率
    timing_info['sharpen'] = time.time() - start_time

    # start_time = time.time()
    # image = apply_denoise(image, prob=0.2)      # 20% 機率
    # timing_info['denoise'] = time.time() - start_time

    start_time = time.time()
    image = apply_blur(image, prob=0.15)        # 15% 機率
    timing_info['blur'] = time.time() - start_time

    start_time = time.time()
    image = apply_noise(image, prob=0.15)       # 15% 機率
    timing_info['noise'] = time.time() - start_time

    # 打印時間統計（如果啟用）
    if print_timing:
        total_time = sum(timing_info.values())
        print(f"\n📊 Augmentation Timing (Total: {total_time*1000:.2f}ms):")
        for aug_name, aug_time in timing_info.items():
            print(f"  - {aug_name:15s}: {aug_time*1000:6.2f}ms ({aug_time/total_time*100:5.1f}%)")

    return image


# ============================================================================
# 👇 整合到 YOLO（不需要修改）
# ============================================================================

from ultralytics.data import YOLODataset
from ultralytics.data.augment import LetterBox


class DynamicAugmentationDataset(YOLODataset):
    """
    自定義 YOLO Dataset，支援動態增強
    """
    def __init__(self, *args, augmentation_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_func = augmentation_func

        if self.augmentation_func:
            print("✓ 啟用動態自定義增強（每個 epoch 都會隨機變化）")

    def load_image(self, i, rect_mode=False):
        """重載 load_image 方法以應用自定義增強"""
        # 呼叫原始的 load_image
        im, (h0, w0), (h, w) = super().load_image(i, rect_mode)

        # 如果有自定義增強函數，應用它
        if self.augmentation_func and self.augment:  # 只在訓練時應用
            try:
                im = self.augmentation_func(im)
            except Exception as e:
                print(f"⚠ 增強失敗: {e}")

        return im, (h0, w0), (h, w)


def setup_dynamic_augmentation(augmentation_func=None):
    """
    設置動態增強（必須在導入 YOLO 之前調用！）

    Args:
        augmentation_func: 自定義增強函數，如果為 None 則使用默認的 apply_custom_augmentation

    Returns:
        setup_complete: bool
    """
    if augmentation_func is None:
        augmentation_func = apply_custom_augmentation

    # 先導入必要模塊以確保它們存在
    import ultralytics.data
    import ultralytics.engine.trainer

    # 保存原始函數
    from ultralytics.data import build_yolo_dataset
    original_build = build_yolo_dataset

    def custom_build_yolo_dataset(*args, **kwargs):
        """自定義的 dataset builder"""
        dataset = original_build(*args, **kwargs)

        # 將增強函數注入到 dataset
        if hasattr(dataset, '__class__'):
            dataset.augmentation_func = augmentation_func
            # 動態修改類別（保留原有功能，添加我們的增強）
            original_load_image = dataset.load_image

            def custom_load_image(i, rect_mode=False):
                """自定義的 load_image 方法"""
                im, (h0, w0), (h, w) = original_load_image(i, rect_mode)

                # 只在訓練時應用增強
                if hasattr(dataset, 'augment') and dataset.augment:
                    try:
                        im = augmentation_func(im)
                    except Exception as e:
                        print(f"⚠ 增強失敗: {e}")

                return im, (h0, w0), (h, w)

            # 替換 load_image 方法
            dataset.load_image = custom_load_image

        return dataset

    # Patch 所有可能的位置
    ultralytics.data.build_yolo_dataset = custom_build_yolo_dataset
    ultralytics.engine.trainer.build_yolo_dataset = custom_build_yolo_dataset

    # 也 patch sys.modules 中的引用
    import sys
    if 'ultralytics.data' in sys.modules:
        sys.modules['ultralytics.data'].build_yolo_dataset = custom_build_yolo_dataset
    if 'ultralytics.engine.trainer' in sys.modules:
        sys.modules['ultralytics.engine.trainer'].build_yolo_dataset = custom_build_yolo_dataset

    print("✓ 動態增強設置完成（已 patch 所有相關模塊）")
    return True


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == '__main__':
    print("這個檔案定義了動態增強函數")
    print("使用方式：在訓練腳本中 import 這個模組")
    print("\n範例:")
    print("  from dynamic_augmentation import setup_dynamic_augmentation")
    print("  setup_dynamic_augmentation()  # 啟用動態增強")
    print("  model.train(...)  # 正常訓練")
