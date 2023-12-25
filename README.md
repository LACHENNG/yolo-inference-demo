## 本项目搜集了各种检测模型的onnx格式推理的代码

### 1. Python-demo

`python-demo-yolov8` 文件夹：展示了使用yolov8训练导出的onnx文件进行目标检测推理的示例代码。


​	model.rar：包含了yolv8的检测模型，检测机器上工作的工人的demo

​    使用前解压model.rar到统计文件夹

​     usage.py中展示三种推理接口的使用


### 2. Cpp-demo

#### 2.1 `linux-platform`
改子文件夹下提供了CMake工程文件，展示了yolov8（或v5）的检测模型推理，使用**opencv dnn** 模块进行推理

#### 2.2 windows-platform
提供了两种推理方式`opencv dnn`` 和`` onnxruntime`（推荐）



**注意**：有些库文件被压缩成了rar，这是为了防止单个文件超限无法上传的情况，git clone 后请手动检查项目目录下的所有压缩包文件，并解压。
