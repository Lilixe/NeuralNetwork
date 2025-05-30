#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
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

    // Skip header row
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

        // Only add if we have exactly 784 pixels
        if (cpt == 784)
        {
            dataset.push_back(img);
        }
        else
        {
            cerr << "Skipping row with " << cpt << " pixels\n";
        }
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

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> initParameters()
{

    int n = 784;                                       // Number of input features (28x28 pixels)
    int m = 10;                                        // Number of output classes (0-9 digits)
    Tensor2D b1 = Tensor2D(1, m), b2 = Tensor2D(1, m); // Biases for the layers
    Tensor2D w1 = Tensor2D(n, m), w2 = Tensor2D(m, m); // Weights for the layers

    // Initialize biases and weights (example values, should be set according to your model)

    // Random number generator for initializing weights and biases
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.5, 0.5);

    // Fill biases and weights with random values
    // Fill biases with random values
    for (int j = 0; j < m; ++j)
    {
        b1(0, j) = dist(gen);
        b2(0, j) = dist(gen);
    }
    // Fill weights with random values
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
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

// Function to perform forward propagation for the first layer
Tensor2D forwardPropagationFL(const Tensor2D &input, const Tensor2D &weights, const Tensor2D &biases)
{
    Tensor2D result = Tensor2D(input.rows(), weights.cols());
    for (int i = 0; i < result.rows(); ++i) // Iterate over each row of the input
    {
        for (int j = 0; j < result.cols(); ++j) // Iterate over each column of the weights
        {
            result(i, j) = biases(0, j);           // Start with bias
            for (int k = 0; k < input.cols(); ++k) // Iterate over each column of the input and row of the weights
            {
                result(i, j) += input(i, k) * weights(k, j); // Add weighted inputs
            }
            if (result(i, j) < 0) // Apply ReLU activation function
            {
                result(i, j) = 0; // Set negative values to zero
            }
        }
    }
}
// Function to forward propagate through the second layer
Tensor2D forwardPropagationSL(const Tensor2D &input, const Tensor2D &weights, const Tensor2D &biases)
{
    float sum = 0.0f; // Initialize sum for the output
    Tensor2D result = Tensor2D(input.rows(), weights.cols());
    for (int i = 0; i < result.rows(); ++i) // Iterate over each row of the input
    {
        for (int j = 0; j < result.cols(); ++j) // Iterate over each column of the weights
        {
            result(i, j) = biases(0, j);           // Start with bias
            for (int k = 0; k < input.cols(); ++k) // Iterate over each column of the input and row of the weights
            {
                result(i, j) += input(i, k) * weights(k, j); // Add weighted inputs
            }
            sum += exp(result(i, j)); // Accumulate the sum for softmax
        }
    }
    // Apply softmax activation function
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = exp(result(i, j)) / sum; // Apply softmax
        }
    }
}

void iterateForwardPropagation(vector<DigitImage> &data, const Tensor2D &w1, const Tensor2D &b1, const Tensor2D &w2, const Tensor2D &b2)
{
    for (DigitImage &img : data)
    {
        // Forward propagate through the first layer
        Tensor2D layer1 = forwardPropagationFL(img.pixels, w1, b1);
        // Forward propagate through the second layer
        Tensor2D output = forwardPropagationSL(layer1, w2, b2);

        img.pixels.copy(output); // Store the output in the image pixels (or handle it as needed)
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
    // displayImageInfo(trainData[0]);

    // Initialize parameters for the neural network
    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = initParameters();

    return 0;
}
