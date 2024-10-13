#include <iostream>
#include <vector>
#include <string>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

using namespace std;

class Inference
{
public:
    Ort::Env env;         // Object managing the resources
    Ort::Session session; // Object representing the model inference

    // Constructor that initializes Ort::Env and Ort::Session ( API version [20] )
    /***  Only API versions [1, 19] are supported in this build ***/
    Inference(const string &modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "MNIST_Inference"),
          session(env, modelPath.c_str(), Ort::SessionOptions{nullptr}) {}

    // Preprocess function
    void preprocess(const cv::Mat &inputImage, vector<float> &outputData)
    {
        // Resize image to 28x28 (for MNIST)
        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(28, 28));

        // Convert image to grayscale and normalize
        cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2GRAY);
        resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255);

        // Flatten image data into a vector
        outputData.assign((float *)resizedImage.datastart, (float *)resizedImage.dataend);
    }

    // Inference function
    int inference(const string &imagePath)
    {
        // Load the image
        cv::Mat inputImage = cv::imread(imagePath);
        if (inputImage.empty())
        {
            cerr << "Could not read the image." << endl;
            return -1;
        }

        // Preprocess the image
        vector<float> inputTensorValues;
        preprocess(inputImage, inputTensorValues);

        // Create input tensor
        vector<int64_t> inputShape = {1, 1, 28, 28}; // (batch_size, channels, height, width)
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size());

        // Run the inference
        vector<const char *> inputNames = {"input"};   // Input name in the model
        vector<const char *> outputNames = {"output"}; // Output name in the model
        vector<Ort::Value> outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            1,
            outputNames.data(),
            1);

        // Extract output data
        float *outputArr = outputTensors[0].GetTensorMutableData<float>();
        vector<float> output(outputArr, outputArr + 10); // Assuming 10 classes for digits 0-9

        // Find the class with the highest probability
        auto maxElement = max_element(output.begin(), output.end());
        int predictedClass = distance(output.begin(), maxElement);

        return predictedClass;
    }
};

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <../mnist_model.onnx> <../four.png>" << endl;
        return 1;
    }

    const string modelPath = argv[1];
    cout << "Model path: " << modelPath << endl;
    const string imagePath = argv[2];
    cout << "Image path: " << imagePath << endl;

    // Initialize the inference object with the model path
    Inference infer(modelPath);

    // Run inference
    int predictedClass = infer.inference(imagePath);

    if (predictedClass != -1)
    {
        cout << "Predicted class: " << predictedClass << endl;
    }

    return 0;
}
