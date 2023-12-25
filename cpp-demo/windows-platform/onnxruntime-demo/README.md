# yolov8/yolov5 Inference C++

### 一、快速体验(以一个手掌打开和关闭的手势检测作为示例)

#### 1.windows平台
打开CMake(cmake-gui) ，将当前目录作为source code目录，并在当前目录的build下构建，依次点击`Configure`` 和 `Generate`` ，然后点击 `Open Project`` 打开visual studio项目，切换到Debug构建，编译运行
#### 2.linux平台
 TODO: 待更新


### 二、推理你自己训练的模型
#### 2.1 总体步骤如下：
1. 训练你的模型，然后导出onnx文件
2. 将导出的文件拷贝到data目录下，假设文件名为myModel.onnx
3. 修改CMakeList.txt 最后的几行中的**your_model.onnx**为你的模型名字
```CMake
configure_file(data/your_model.onnx ${CMAKE_CURRENT_BINARY_DIR}/model.onnx COPYONLY)
```
4. 使用CMake工具执行构建（windows上可以使用CMake gui工具，参考前面的**快速体验**）

#### 2.2 详细步骤
1. 对于模型**训练和导出**，训练可以参考 Github中ultralytics项目的yolov8仓库，官方有详细说明，不再赘述，这里主要讲一下如何讲训练得到的.pt模型文件转成onnx格式。注意，torch不要用太高的版本，否则会出现奇怪的不兼容和推理结果错误的问题，难以排查，这里我测试通过的导出对于的核心库极版本为：
```
$ pip list | grep torch 
torch                   1.11.0
torchvision             0.12.0
``` 
或者创建一个我测试通过的版本环境（仅仅测试了Linux下）
```
# 创建一个conda虚拟环境
conda env create -n "yolo" python=3.10 

# 激活
conda activate yolo

# 安装软件包用于导出onnx
pip install -r requirement.txt

```
**执行导出** （注意安装对应的环境以便能识别yolo命令)

To export yolov8 models:

```
yolo export \
model=yolov8s.pt \
imgsz=[480,640] \
format=onnx \
opset=12
```

To export yolov5 models:

```
python3 export.py \
--weights yolov5s.pt \
--img 480 640 \
--include onnx \
--opset 12
```

其他详细步骤省略