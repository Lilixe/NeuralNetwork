#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include "tensor.h"
#include "config.h"

struct DigitImage {
    int label;
    Tensor2D pixels;
    DigitImage(int l, int n, int m) : label(l), pixels(n, m) {}
};

inline std::vector<DigitImage> loadMNIST(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<DigitImage> dataset;
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return dataset;
    }
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string val;
        if (!std::getline(ss, val, ',')) continue;
        int label;
        try { label = std::stoi(val); } catch (...) { continue; }

        DigitImage img(label, 1, INPUT_SIZE);
        int cpt = 0;
        while (std::getline(ss, val, ',') && cpt < img.pixels.size())
        {
            try { img.pixels(0, cpt) = std::stof(val) / 255.0f; }
            catch (...) { img.pixels(0, cpt) = 0.0f; }
            ++cpt;
        }
        dataset.push_back(img);
    }
    return dataset;
}

inline void randomizeVect(std::vector<DigitImage> &data)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

inline std::pair<std::vector<DigitImage>, std::vector<DigitImage>>
splitData(const std::vector<DigitImage> &data)
{
    int idx = static_cast<int>(data.size() * 0.8);
    return { {data.begin(), data.begin() + idx},
             {data.begin() + idx, data.end()} };
}

// Returns (labels[1×N], pixels[784×N])
inline std::pair<Tensor2D, Tensor2D> toMatrix(std::vector<DigitImage> &data)
{
    int cols = static_cast<int>(data.size());
    Tensor2D pixels(INPUT_SIZE, cols), labels(1, cols);
    for (int i = 0; i < cols; ++i)
    {
        labels(0, i) = static_cast<float>(data[i].label);
        for (int j = 0; j < INPUT_SIZE; ++j)
            pixels(j, i) = data[i].pixels(0, j);
    }
    return std::make_pair(labels, pixels);
}

struct DataSet {
    Tensor2D trainLabels, trainPixels;
    Tensor2D devLabels,   devPixels;
};

inline DataSet initData(std::vector<DigitImage> data)
{
    randomizeVect(data);
    std::vector<DigitImage> trainVec, devVec;
    std::tie(trainVec, devVec) = splitData(data);

    DataSet ds;
    std::tie(ds.trainLabels, ds.trainPixels) = toMatrix(trainVec);
    std::tie(ds.devLabels,   ds.devPixels)   = toMatrix(devVec);
    return ds;
}
