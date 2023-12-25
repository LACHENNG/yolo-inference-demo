#include "inference.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;
 
bool Yolov8Onnx::ReadModel(const std::string &modelPath, bool isCuda, int cudaId, bool warmUp){
    if (_batchSize < 1) _batchSize =1;
    try
    {
        std::vector<std::string> available_providers = GetAvailableProviders();
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
 
 
        if (isCuda && (cuda_available == available_providers.end()))
        {
            std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        else if (isCuda && (cuda_available != available_providers.end()))
        {
            std::cout << "************* Infer model on GPU! *************" << std::endl;
//#if ORT_API_VERSION < ORT_OLD_VISON
//          OrtCUDAProviderOptions cudaOption;
//          cudaOption.device_id = cudaID;
//            _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
//#else
//          OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
//#endif
        }
        else
        {
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        //
 
        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
 
#ifdef _WIN32
        std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif
 
        Ort::AllocatorWithDefaultOptions allocator;
        //init input
        _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
        _inputName = _OrtSession->GetInputName(0, allocator);
        _inputNodeNames.push_back(_inputName);
#else
        _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
        _inputNodeNames.push_back(_inputName.get());
#endif
        //cout << _inputNodeNames[0] << endl;
        Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputNodeDataType = input_tensor_info.GetElementType();
        _inputTensorShape = input_tensor_info.GetShape();
 
        if (_inputTensorShape[0] == -1)
        {
            _isDynamicShape = true;
            _inputTensorShape[0] = _batchSize;
 
        }
        if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
            _isDynamicShape = true;
            _inputTensorShape[2] = _netHeight;
            _inputTensorShape[3] = _netWidth;
        }
        //init output
        _outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
        _output_name0 = _OrtSession->GetOutputName(0, allocator);
        _outputNodeNames.push_back(_output_name0);
#else
        _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
        _outputNodeNames.push_back(_output_name0.get());
#endif
        Ort::TypeInfo type_info_output0(nullptr);
        type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
 
        auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
        _outputNodeDataType = tensor_info_output0.GetElementType();
        _outputTensorShape = tensor_info_output0.GetShape();
 
        //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
        //_outputMaskTensorShape = tensor_info_output1.GetShape();
        //if (_outputTensorShape[0] == -1)
        //{
        //  _outputTensorShape[0] = _batchSize;
        //  _outputMaskTensorShape[0] = _batchSize;
        //}
        //if (_outputMaskTensorShape[2] == -1) {
        //  //size_t ouput_rows = 0;
        //  //for (int i = 0; i < _strideSize; ++i) {
        //  //  ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
        //  //}
        //  //_outputTensorShape[1] = ouput_rows;
 
        //  _outputMaskTensorShape[2] = _segHeight;
        //  _outputMaskTensorShape[3] = _segWidth;
        //}
 
        //warm up
        if (isCuda && warmUp) {
            //draw run
            cout << "Start warming up" << endl;
            size_t input_tensor_length = VectorProduct(_inputTensorShape);
            float* temp = new float[input_tensor_length];
            std::vector<Ort::Value> input_tensors;
            std::vector<Ort::Value> output_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                _inputTensorShape.size()));
            for (int i = 0; i < 3; ++i) {
                output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                    _inputNodeNames.data(),
                    input_tensors.data(),
                    _inputNodeNames.size(),
                    _outputNodeNames.data(),
                    _outputNodeNames.size());
            }
 
            delete[]temp;
        }
    }
    catch (const std::exception&) {
        return false;
    }
    return true;
 
}
 
int Yolov8Onnx::Preprocessing(const std::vector<cv::Mat> &SrcImgs,
                              std::vector<cv::Mat> &OutSrcImgs,
                              std::vector<cv::Vec4d> &params){
    OutSrcImgs.clear();
    Size input_size = Size(_netWidth, _netHeight);
 
    // 信封处理
    for (size_t i=0; i<SrcImgs.size(); ++i){
        Mat temp_img = SrcImgs[i];
        Vec4d temp_param = {1,1,0,0};
        if (temp_img.size() != input_size){
            Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            OutSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        }
        else {
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
 
    int lack_num = _batchSize - SrcImgs.size();
    if (lack_num > 0){
        Mat temp_img = Mat::zeros(input_size, CV_8UC3);
        Vec4d temp_param = {1,1,0,0};
        OutSrcImgs.push_back(temp_img);
        params.push_back(temp_param);
    }
    return 0;
}
 
bool Yolov8Onnx::OnnxBatchDetect(std::vector<cv::Mat> &srcImgs, std::vector<std::vector<OutputDet> > &output)
{
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
 
    //preprocessing (信封处理)
    Preprocessing(srcImgs, input_images, params);
    // [0~255] --> [0~1]; BGR2RGB
    Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0,0,0), true, false);
 
    // 前向传播得到推理结果
    int64_t input_tensor_length = VectorProduct(_inputTensorShape);// ?
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data,
                                                            input_tensor_length, _inputTensorShape.data(),
                                                            _inputTensorShape.size()));
 
    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );
 
    //post-process
    int net_width = _className.size() + 4;
    float* all_data = output_tensors[0].GetTensorMutableData<float>(); // 第一张图片的输出
 
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // 一张图片输出的维度信息 [1, 84, 8400]
 
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0]; // 一张图片输出所占内存长度 8400*84
 
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index){
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t(); // [1, 84 ,8400] -> [1, 8400, 84]
 
        all_data += one_output_length; //指针指向下一个图片的地址
 
        float* pdata = (float*)output0.data; // [x,y,w,h,class1,class2.....class80]
        int rows = output0.rows; // 预测框的数量 8400
 
        // 一张图片的预测框
        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;
        for (int r=0; r<rows; ++r){//如果是yolov5则需要做修改
            Mat scores(1, _className.size(), CV_32F, pdata + 4); // 80个类别的概率
 
            // 得到最大类别概率、类别索引
            Point classIdPoint;
            double max_class_soces;
            minMaxLoc(scores, 0, &max_class_soces, 0, &classIdPoint);
            max_class_soces = (float)max_class_soces;
 
            // 预测框坐标映射到原图上
            if (max_class_soces >= _classThreshold){
                // rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0]; //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1]; //y
                float w = pdata[2] / params[img_index][0]; //w
                float h = pdata[3] / params[img_index][1]; //h
 
                int left = MAX(int(x - 0.5 *w +0.5), 0);
                int top = MAX(int(y - 0.5*h + 0.5), 0);
 
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_soces);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width; //下一个预测框
        }
 
        // 对一张图的预测框执行NMS处理
        vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThrehold, nms_result); // 还需要classThreshold？
 
        // 对一张图片：依据NMS处理得到的索引，得到类别id、confidence、box，并置于结构体OutputDet的容器中
        vector<OutputDet> temp_output;
        for (size_t i=0; i<nms_result.size(); ++i){
            int idx = nms_result[i];
            OutputDet result;
            result.id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            temp_output.push_back(result);
        }
        output.push_back(temp_output); // 多张图片的输出；添加一张图片的输出置于此容器中
 
    }
    if (output.size()) {
        return true;
    }
    else {
        return false;
    }
 
}
 
 
bool Yolov8Onnx::OnnxDetect(cv::Mat &srcImg, std::vector<OutputDet> &output){
    vector<Mat> input_data = {srcImg};
    vector<vector<OutputDet>> temp_output;
 
    if(OnnxBatchDetect(input_data, temp_output)){
        output = temp_output[0];
        return true;
    }
    else return false;
}
 