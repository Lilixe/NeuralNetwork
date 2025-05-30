#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include "tensor.cpp"

using namespace std;

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
