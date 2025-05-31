# NeuralNetwork

A simple neural network implementation in C++ designed for educational purposes.  
This project demonstrates core neural network concepts such as forward propagation, backpropagation, and training on the MNIST dataset.

## 🧠 Features

- ✅ Feedforward Neural Network
- 🔁 Backpropagation algorithm
- 🖼️ MNIST dataset loading and training
- 🧪 Written in C++

## 📦 Project Structure

NeuralNetwork/
├── cppfiles/ # C++ source code files
├── .vscode/ # VS Code config (optional)
├── ressources.txt # Reference links or notes
└── main.cpp # Entry point

## 🚀 Getting Started

### Prerequisites

- A C++11-compatible compiler (e.g., `g++`)
- [CMake](https://cmake.org/) (optional but recommended)
- MNIST dataset in CSV format (e.g., `mnist_train.csv`, `mnist_test.csv`)

### 🔨 Build Instructions

#### Option 1: Compile Manually

bash
g++ -std=c++11 -o neural_network cppfiles/*.cpp main.cpp

#### Option 2: Using CMake

mkdir build
cd build
cmake ..
make

#### ▶️ Run

bash
./neural_network

✍️ Author
github.com/Lilixe
Built for learning and exploration. Not intended for production use.

