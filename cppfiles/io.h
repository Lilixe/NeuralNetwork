#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include "tensor.h"
#include "nn_model.h"
#include "config.h"

inline void saveParameters(const NetworkParams &p, const std::string &filename)
{
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open file for saving: " << filename << "\n";
        return;
    }
    std::cout << "Saving parameters to " << filename << "...\n";

    auto save = [&](const Tensor2D &t) {
        for (int i = 0; i < t.rows(); ++i)
            for (int j = 0; j < t.cols(); ++j) {
                float v = t(i, j);
                f.write(reinterpret_cast<const char *>(&v), sizeof(float));
            }
    };
    save(p.b1); save(p.b2); save(p.w1); save(p.w2);
    std::cout << "Parameters saved.\n";
}

inline NetworkParams loadParameters(const std::string &filename)
{
    NetworkParams p;
    p.b1 = Tensor2D(HIDDEN_SIZE, 1);
    p.b2 = Tensor2D(OUTPUT_SIZE, 1);
    p.w1 = Tensor2D(HIDDEN_SIZE, INPUT_SIZE);
    p.w2 = Tensor2D(OUTPUT_SIZE, HIDDEN_SIZE);

    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << "\n";
        return p;
    }
    std::cout << "Loading parameters from " << filename << "...\n";

    auto load = [&](Tensor2D &t) {
        for (int i = 0; i < t.rows(); ++i)
            for (int j = 0; j < t.cols(); ++j)
                if (!f.read(reinterpret_cast<char *>(&t(i, j)), sizeof(float)))
                    std::cerr << "Read error in parameter file.\n";
    };
    load(p.b1); load(p.b2); load(p.w1); load(p.w2);
    return p;
}
