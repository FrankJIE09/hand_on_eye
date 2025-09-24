# 梅卡相机内参标定工具

本工具用于梅卡相机的内参标定，使用5x4标定板（50mm间距）。

## 文件说明

- `step1_camera_capture.py` - 梅卡相机拍照脚本
- `step2_image_processing.py` - 图像处理和角点检测
- `step3_calibration.py` - 相机内参标定
- `utils.py` - 工具函数（来自Orbbec SDK）

## 使用步骤

### 1. 拍照
```bash
python step1_camera_capture.py
```
选择拍照模式：
- 交互式拍照：手动按's'键拍照
- 自动拍照：自动拍摄多张图像

拍摄的图像会保存在 `./captured_images/` 目录中。

### 2. 图像处理
```bash
python step2_image_processing.py
```
处理拍摄的图像，检测标定板角点，生成处理结果。

### 3. 内参标定
```bash
python step3_calibration.py
```
计算相机内参，输出标定结果。

## 输出文件

- `camera_intrinsics.yaml` - YAML格式的相机内参
- `camera_intrinsics_results.txt` - 文本格式的标定结果
- `processed_images/` - 处理后的图像（带角点标记）
- `processing_data/` - 处理数据文件
- `excel_results/` - Excel格式的详细结果

## 标定板规格

- 尺寸：5x4 圆点
- 间距：50mm
- 类型：非对称圆点网格

## 环境要求

- Python 3.x
- OpenCV
- NumPy
- Pandas
- PyYAML
- 梅卡相机SDK (mecheye)

## 注意事项

1. 确保梅卡相机已正确连接
2. 标定板需要平整放置
3. 拍摄时保持相机稳定
4. 建议拍摄15-20张不同角度的图像
5. 确保标定板在图像中完整可见
