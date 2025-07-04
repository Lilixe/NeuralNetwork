#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <thread>

#include "tensor.cpp"
#include <mutex>
using namespace std;

// All Matrix multiplication will be done using the naive approch which is enough for this project. With a complexity of O(n^3) for n x n matrices.
// The project is a simple neural network implementation for MNIST digit classification.

/**************************************************
 *                INITIALIZATION                  *
 * This section initializes the neural network,   *
 * including weights, biases, and structure.      *
 **************************************************/

// Define a structure to hold the digit image data until processed to matrix format
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
                img.pixels(0, cpt) = stof(value) / 255.0f; // Normalize pixel value to [0, 1]
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

    int input_size = 784;                                                                     // Number of input features (28x28 pixels)
    int hidden_size = 128;                                                                    // Number of hidden neurons
    int output_size = 10;                                                                     // Number of output classes (0-9 digits)
    Tensor2D b1 = Tensor2D(hidden_size, 1), b2 = Tensor2D(output_size, 1);                    // Biases for the layers
    Tensor2D w1 = Tensor2D(hidden_size, input_size), w2 = Tensor2D(output_size, hidden_size); // Weights for the layers

    // Initialize biases and weights (example values, should be set according to your model)

    // Random number generator for initializing weights and biases
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.5, 0.5);

    // Fill biases and weights with random values
    // Fill biases with random values
    // Fill biases with random values
    for (int i = 0; i < hidden_size; ++i)
    {
        b1(i, 0) = dist(gen);
    }
    for (int i = 0; i < output_size; ++i)
    {
        b2(i, 0) = dist(gen);
    }
    // Fill weights with random values
    for (int i = 0; i < hidden_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            w1(i, j) = dist(gen);
        }
    }
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < hidden_size; ++j)
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
    return make_pair(resultLbl, resultPxl); // Return the matrix representation of the dataset
}

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> initData(vector<DigitImage> trainData)
{
    // Randomize the training data
    trainData = randomizeVect(trainData);
    vector<DigitImage> devData;

    // Split the training data into training and development sets
    tie(trainData, devData) = splitData(trainData);

    Tensor2D trainLabels, trainPixels, devLabels, devPixels;
    tie(trainLabels, trainPixels) = toMatrix(trainData); // Convert training data to matrix format
    tie(devLabels, devPixels) = toMatrix(devData);       // Convert development data to matrix format

    return make_tuple(trainLabels, trainPixels, devLabels, devPixels); // Return the matrices for training and development sets
}

/**************************************************
 *             FORWARD PROPAGATION                *
 * Computes the output of the network by passing  *
 * inputs through each layer using activation     *
 * functions (e.g., ReLU, softmax).               *
 **************************************************/

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
    for (int j = 0; j < t.cols(); ++j)
    {
        float max = t.maxCol(j); // Find the maximum value in the column for numerical stability
        float sum = 0.0f;        // Initialize sum for softmax
        for (int i = 0; i < t.rows(); ++i)
        {
            result(i, j) = exp(t(i, j) - max); // Subtract max for numerical stability
            sum += result(i, j);               // Accumulate the sum for softmax
        }
        for (int i = 0; i < result.rows(); ++i)
        {
            result(i, j) /= sum; // Normalize by the sum
        }
    }
    return result; // Return the softmax result
}

// Function to perform forward propagation for the first layer
Tensor2D forwardPropagationLayer(const Tensor2D input, const Tensor2D weights, const Tensor2D biases)
{
    Tensor2D result = weights.dot(input) + biases; // Perform matrix multiplication and add biases
    return result;                                 // Return the result of the forward propagation
}

// Function to iterate over the dataset and perform forward propagation
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> ForwardPropagation(
    Tensor2D &data, const Tensor2D &w1, const Tensor2D &b1, const Tensor2D &w2, const Tensor2D &b2)
{
    // Vectors to hold the outputs of each layer and the updated images
    Tensor2D L1output, L1outputRelu, L2output, softmaxOutput;

    // Forward propagate through the first layer
    L1output = forwardPropagationLayer(data, w1, b1);
    L1outputRelu = relu(L1output); // Apply ReLU activation function

    // Forward propagate through the second layer
    L2output = forwardPropagationLayer(L1outputRelu, w2, b2);
    softmaxOutput = softmax(L2output); // Apply softmax activation function

    return make_tuple(L1output, L1outputRelu, L2output, softmaxOutput); // Return the tuple of vectors
}

/**************************************************
 *             BACKWARD PROPAGATION                *
 * Computes gradients and updates weights and      *
 * biases based on the loss function.              *
 **************************************************/

// Function to compute the derivative of ReLU activation function
Tensor2D derivativeRelu(const Tensor2D &t, const Tensor2D variator)
{
    Tensor2D result = Tensor2D(t.rows(), t.cols()); // Create a new tensor for the result
    for (int i = 0; i < t.rows(); ++i)
    {
        for (int j = 0; j < t.cols(); ++j)
        {
            result(i, j) = (variator(i, j) > 0) ? t(i, j) : 0.0f; // Derivative of ReLU is the value of t if variator > 0, else 0
        }
    }
    return result; // Return the derivative of ReLU
}

// Function to convert labels to one-hot encoding tensor
Tensor2D oneHot(const Tensor2D labels)
{
    Tensor2D result = Tensor2D(labels.cols(), 10); // Create a new tensor for the result with 10 classes
    for (int i = 0; i < labels.cols(); ++i)
    {
        result(i, labels(0, i)) = 1.0f; // Set the corresponding class to 1
    }
    return result.transpose(); // Return the one-hot encoded tensor
}

// Function to perform backward propagation
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> backwardPropagation(Tensor2D &w2, Tensor2D &l1, Tensor2D &l1relu, Tensor2D &l2, Tensor2D &Pixels, Tensor2D &forwardResults, Tensor2D &oneHotLabels)
{
    int m = forwardResults.size();         // Get the number of samples (columns in the one-hot labels)
    Tensor2D dl1, dw1, db1, dl2, dw2, db2; // Gradients for the layers

    // Compute the gradient of the loss with respect to the second layer
    dl2 = forwardResults - oneHotLabels;            // Compute the gradient of the loss with respect to the output layer
    dw2 = dl2.dot(l1relu.transpose()) * (1.0f / m); // Compute the gradient of the weights for the second layer
    Tensor2D dEl2 = dl2.sumColumn();
    db2 = dEl2 * (1.0f / m); // Compute the gradient of the biases for the second layer

    // Compute the gradient of the loss with respect to the first layer
    dl1 = derivativeRelu(w2.transpose().dot(dl2), l1); // Apply the derivative of ReLU to the gradient of the first layer
    dw1 = dl1.dot(Pixels.transpose()) * (1.0f / m);    // Compute the gradient of the weights for the first layer
    Tensor2D dEl1 = dl1.sumColumn();
    db1 = dEl1 * (1.0f / m); // Compute the gradient of the biases for the first layer

    return make_tuple(dw1, db1, dw2, db2);
}

/***************************************
 *     NEURAL NET PARAMETER UPDATE     *
 *  Applies gradient-based updates to  *
 *  weights and biases after backprop. *
 ***************************************/

// Function to calculate the accuracy of predictions against labels
float accuracy(const Tensor2D &predictions, const Tensor2D &labels)
{
    float correct = 0;
    for (int i = 0; i < predictions.cols(); ++i)
    {
        // Use maxCol to get the predicted class for column i
        float pred_class = predictions.maxColIdx(i);
        float true_class = (labels(0, i));
        if (pred_class == true_class)
        {
            ++correct;
        }
    }
    float acc = (correct / predictions.cols()) * 100.0f;
    return acc;
}

// Function to update the parameters (weights and biases) of the neural network
void updateParameters(Tensor2D &w1, Tensor2D &b1, Tensor2D &w2, Tensor2D &b2, const Tensor2D &dw1, const Tensor2D &db1, const Tensor2D &dw2, const Tensor2D &db2, float alpha)
{
    // Update weights and biases using gradient descent
    w1 = w1 - (dw1 * alpha); // Update weights for the first layer
    b1 = b1 - (db1 * alpha); // Update biases for the first layer
    w2 = w2 - (dw2 * alpha); // Update weights for the second layer
    b2 = b2 - (db2 * alpha); // Update biases for the second layer
}

/***********************************************
 *         MULTITHREADING SUPPORT               *
 * Use threading to parallelize training.       *
 ***********************************************/

// Function to perform gradient descent in parallel using threads
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> threadedGradientDescent(Tensor2D labels, Tensor2D pixels, int iterations, float alpha, int numThreads)
{
    // Initialize parameters for the neural network
    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = initParameters();
    Tensor2D oneHotLabels = oneHot(labels);

    int batch_size = pixels.cols() / numThreads;

    // Declare mutex outside the lambda so all threads share it
    static mutex mtx;

    // Lambda function to perform training on a batch of data
    auto train_batch = [&](int start, int end)
    {
        Tensor2D l1, l1relu, l2, l2softmax;
        Tensor2D dw1, db1, dw2, db2;
        for (int i = 0; i < iterations; ++i)
        {
            // Slice the batch of pixels and labels for the current thread
            Tensor2D batchPixels = pixels.sliceCols(start, end);
            Tensor2D batchLabels = labels.sliceCols(start, end);
            Tensor2D batchOneHot = oneHot(batchLabels);

            // Forward propagation
            tie(l1, l1relu, l2, l2softmax) = ForwardPropagation(batchPixels, w1, b1, w2, b2);

            // Backward propagation
            tie(dw1, db1, dw2, db2) = backwardPropagation(w2, l1, l1relu, l2, batchPixels, l2softmax, batchOneHot);

            // Lock for parameter update (shared mutex)
            mtx.lock();
            updateParameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha);
            mtx.unlock();
        }
    };

    // Create threads for parallel training
    vector<thread> threads;
    for (int t = 0; t < numThreads; ++t)
    {
        int start = t * batch_size;
        int end = (t == numThreads - 1) ? pixels.cols() : (t + 1) * batch_size;
        threads.emplace_back(train_batch, start, end);
    }
    for (auto &th : threads)
        th.join();

    return make_tuple(b1, b2, w1, w2);
}

/*****************************
 *       MAIN FUNCTION       *
 * Program execution starts. *
 *****************************/

int main()
{
    // Load MNIST dataset
    vector<DigitImage> trainData = loadMNIST("../mnist_train.csv");
    vector<DigitImage> testData = loadMNIST("../mnist_test.csv");

    // convert all datasets to matrix format
    Tensor2D trainLabels, trainPixels, devLabels, devPixels, testLabels, testPixels;
    tie(testLabels, testPixels) = toMatrix(testData);    // Convert test data to matrix format
    tie(trainLabels, trainPixels, devLabels, devPixels) = initData(trainData); // Convert training and development data to matrix format

    // Gradient Descent on the training set
    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = threadedGradientDescent(trainLabels, trainPixels, 1000, 0.1f, 10); // Perform gradient descent on the development set

    // Evaluate on development set
    Tensor2D l1, l1relu, l2, predictions;
    tie(l1, l1relu, l2, predictions) = ForwardPropagation(testPixels, w1, b1, w2, b2);

    float testAcc = accuracy(predictions, testLabels);
    cout << "\n==============================\n";
    cout << "Final Accuracy on Test Set: " << testAcc << "%\n";
    cout << "==============================\n";

    return 0;
}