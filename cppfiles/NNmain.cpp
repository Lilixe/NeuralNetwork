#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>

#include "tensor.cpp"

using namespace std;
// All Matrix multiplication will be done using the naive approch which is enough for this project. With a complexity of O(n^3) for n x n matrices.
// The project is a simple neural network implementation for MNIST digit classification.

// Define a structure to hold the digit image data
struct DigitImage
{
    int label;       // Label of the digit (0-9)
    Tensor2D pixels; // 28x28 = 784 pixels

    // Default constructor to initialize pixels with correct size
    DigitImage(int l, int n, int m) : label(l), pixels(n, m) {}
};

// Function to load MNIST dataset from a CSV file
vector<DigitImage> loadMNIST(const string &filename)
{
    ifstream file(filename);
    vector<DigitImage> dataset;

    if (!file.is_open())
    {
        cerr << "Failed to open file.\n";
        return dataset;
    }

    string line;

    // Skip header row because it does not contain data on this training set
    getline(file, line);

    while (getline(file, line))
    {
        stringstream sstream(line);
        string value;

        // Get label
        if (!getline(sstream, value, ','))
            continue;
        int label;
        try
        {
            label = stoi(value);
        }
        catch (...)
        {
            continue; // Skip malformed line
        }

        DigitImage img(label, 1, 784);

        // Get 784 pixels
        int cpt = 0;
        while (getline(sstream, value, ',') && cpt < img.pixels.size())
        {
            try
            {
                img.pixels(0, cpt) = stof(value);
            }
            catch (...)
            {
                img.pixels(0, cpt) = 0.0f;
            }
            ++cpt;
        }
        dataset.push_back(img);
    }

    file.close();
    return dataset;
}

// Function to display image information
void displayImageInfo(const DigitImage &img)
{
    // display the label and pixels of the image
    cout << "Label: " << img.label << "\nPixels:\n";

    // Print the pixels in a 28x28 format
    img.pixels.display();
}

// Function to randomize the dataset
vector<DigitImage> randomizeVect(vector<DigitImage> &dataset)
{
    random_device rd; // Obtain a random number from hardware
    mt19937 g(rd());  // Seed the generator
    shuffle(begin(dataset), end(dataset), g);
    return dataset;
}

// Function to split dataset into training and development sets
pair<vector<DigitImage>, vector<DigitImage>> splitData(vector<DigitImage> &dataset)
{

    // Example: Split 80% train, 20% dev
    int splitIndex = static_cast<size_t>(dataset.size() * 0.8);

    vector<DigitImage> train(dataset.begin(), dataset.begin() + splitIndex);
    vector<DigitImage> dev(dataset.begin() + splitIndex, dataset.end());
    return make_pair(train, dev); // Return a pair of vectors
}

// Function to initialize weights and biases for the neural network
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> initParameters()
{

    int n = 784;                                       // Number of input features (28x28 pixels)
    int m = 10;                                        // Number of output classes (0-9 digits)
    Tensor2D b1 = Tensor2D(m, 1), b2 = Tensor2D(m, 1); // Biases for the layers
    Tensor2D w1 = Tensor2D(m, n), w2 = Tensor2D(m, m); // Weights for the layers

    // Initialize biases and weights (example values, should be set according to your model)

    // Random number generator for initializing weights and biases
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.5, 0.5);

    // Fill biases and weights with random values
    // Fill biases with random values
    for (int j = 0; j < m; ++j)
    {
        b1(j, 0) = dist(gen);
        b2(j, 0) = dist(gen);
    }
    // Fill weights with random values
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            w1(i, j) = dist(gen);
        }
    }
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            w2(i, j) = dist(gen);
        }
    }

    return make_tuple(b1, b2, w1, w2); // Return a tuple of vectors for biases and weights
}

// Function to convert a vector of DigitImage to a matrix format
pair<Tensor2D, Tensor2D> toMatrix(vector<DigitImage> &data)
{
    // Convert the vector of DigitImage to a matrix format
    int rows = 784; // Each image has 784 pixels (28x28)
    int cols = data.size(); 
    Tensor2D resultPxl(rows, cols);
    Tensor2D resultLbl(1, cols); // Create a matrix for labels

    for (int i = 0; i < cols; ++i)
    {
        resultLbl(0, i) = data[i].label; // Fill the label matrix with labels
        for (int j = 0; j < rows; ++j)
        {
            resultPxl(j, i) = data[i].pixels(0, j); // Fill the matrix with pixel values (row: pixel, col: image)
        }
    }
    return make_pair(resultLbl,resultPxl); // Return the matrix representation of the dataset
}

// ReLU activation function
Tensor2D relu(Tensor2D t)
{
    Tensor2D result = Tensor2D(t.rows(), t.cols()); // Create a new tensor for the result
    for (int i = 0; i < t.rows(); ++i)
    {
        for (int j = 0; j < t.cols(); ++j)
        {
            result(i, j) = (t(i, j) < 0) ? 0 : t(i, j);
        }
    }
    return result;
}

// Softmax activation function
Tensor2D softmax(Tensor2D t)
{
    Tensor2D result = Tensor2D(t.rows(), t.cols()); // Create a new tensor for the result
    float max = t.max(); // Find the maximum value in the tensor
    float sum = 0.0f; // Initialize sum for softmax
    for (int i = 0; i < t.rows(); ++i)
    {
        for (int j = 0; j < t.cols(); ++j)
        {
            result(i, j) = exp(t(i, j) - max); // Subtract max for numerical stability
            sum += result(i, j); // Accumulate the sum for softmax
        }
    }
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) /= sum; // Normalize by the sum
        }
    }
    return result; // Return the softmax result
}

// Function to perform forward propagation for the first layer
Tensor2D forwardPropagationFL(const Tensor2D input, const Tensor2D weights, const Tensor2D biases)
{
    Tensor2D result = weights.dot(input) + biases; // Perform matrix multiplication and add biases
    return result; // Return the result of the forward propagation
}

// Function to forward propagate through the second layer
Tensor2D forwardPropagationSL(const Tensor2D input, const Tensor2D weights, const Tensor2D biases)
{
    Tensor2D result = weights.dot(input) + biases; // Perform matrix multiplication and add biases
    return result; // Return the result of the forward propagation
}

// Function to iterate over the dataset and perform forward propagation
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> iterateForwardPropagation(
    Tensor2D &data, const Tensor2D &w1, const Tensor2D &b1, const Tensor2D &w2, const Tensor2D &b2)
{
    // Vectors to hold the outputs of each layer and the updated images
    Tensor2D L1output, L1outputRelu, L2output, softmaxOutput;
    
    // Forward propagate through the first layer
    L1output = forwardPropagationFL(data, w1, b1);
    L1outputRelu = relu(L1output); // Apply ReLU activation function
    
    // Forward propagate through the second layer
    L2output = forwardPropagationSL(L1outputRelu, w2, b2);
    softmaxOutput = softmax(L2output); // Apply softmax activation function
    
    return make_tuple(L1output, L1outputRelu, L2output, softmaxOutput); // Return the tuple of vectors
}

// Function to average the tensor by dividing each element by m
Tensor2D averagedTensor (const Tensor2D &t, int m)
{
    Tensor2D result = Tensor2D(t.rows(), t.cols()); // Create a new tensor for the result
    for (int i = 0; i < t.rows(); ++i)
    {
        for (int j = 0; j < t.cols(); ++j)
        {
            result(i, j) /= m; // Divide each element by m to average
        }
    }
    return result; // Return the averaged tensor
}

// Function to perform backward propagation
void backwardPropagation(vector<Tensor2D> &l1, vector<Tensor2D> &l1relu, vector<Tensor2D> &l2, vector<DigitImage> &forwardResults)
{
    int m = forwardResults[0].pixels.size() * forwardResults.size();
    vector<Tensor2D> dl1relu, dw1, db1, dresults, dw2, db2;
    for (DigitImage &img : forwardResults)
    {
        Tensor2D dt = img.pixels; // Get the output of the second layer
        dt(0, img.label) -= 1;
        dl1relu.push_back(dt);
    }
}
    

int main()
{
    // Load MNIST dataset
    vector<DigitImage> trainData = loadMNIST("../mnist_train.csv");
    vector<DigitImage> testData = loadMNIST("../mnist_test.csv");

    // Randomize the training data
    trainData = randomizeVect(trainData);
    vector<DigitImage> devData;

    
    // Split the training data into training and development sets
    tie(trainData, devData) = splitData(trainData);
    
    Tensor2D trainLabels, trainPixels, devLabels, devPixels, testLabels, testPixels;
    tie(trainLabels, trainPixels) = toMatrix(trainData); // Convert training data to matrix format
    tie(devLabels, devPixels) = toMatrix(devData); // Convert development data to matrix format
    tie(testLabels, testPixels) = toMatrix(testData); // Convert test data to matrix format
    
    // Initialize parameters for the neural network
    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = initParameters();
    
    
    Tensor2D l1, l1relu, l2, l2softmax; // Vectors to hold outputs of the layers results
    tie(l1, l1relu, l2, l2softmax) = iterateForwardPropagation(trainPixels, w1, b1, w2, b2);
    
    // Display the first image's output after forward propagation
    cout << "Output size n : " << l2softmax.rows() << " by m : " << l2softmax.cols() << endl;
    
    return 0;
}
