#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <algorithm>

class Tensor2D
{
private:
    int n, m;
    std::vector<float> data;

    int computeOffset(int i, int j) const
    {
        if (i < 0 || i >= n || j < 0 || j >= m)
            throw std::out_of_range("Index out of bounds");
        return i * m + j;
    }

public:
    Tensor2D() : n(0), m(0) {}

    Tensor2D(int n, int m) : n(n), m(m), data(n * m, 0.0f) {}

    Tensor2D(const Tensor2D &other) = default;
    Tensor2D(Tensor2D &&other) noexcept = default;
    Tensor2D &operator=(const Tensor2D &other) = default;
    Tensor2D &operator=(Tensor2D &&other) noexcept = default;

    float &operator()(int i, int j)
    {
        return data[computeOffset(i, j)];
    }

    float operator()(int i, int j) const
    {
        return data[computeOffset(i, j)];
    }

    // Supports same-shape addition and column-vector broadcast
    Tensor2D operator+(const Tensor2D &other) const
    {
        if (n == other.n && m == other.m)
        {
            Tensor2D result(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    result(i, j) = operator()(i, j) + other(i, j);
            return result;
        }
        if (other.m == 1 && other.n == n)
        {
            Tensor2D result(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    result(i, j) = operator()(i, j) + other(i, 0);
            return result;
        }
        throw std::invalid_argument("Incompatible shapes for addition");
    }

    Tensor2D operator-(const Tensor2D &other) const
    {
        if (n != other.n || m != other.m)
            throw std::invalid_argument("Tensors must have the same dimensions for subtraction");
        Tensor2D result(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                result(i, j) = operator()(i, j) - other(i, j);
        return result;
    }

    Tensor2D operator*(float factor) const
    {
        Tensor2D result(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                result(i, j) = operator()(i, j) * factor;
        return result;
    }

    int rows() const { return n; }
    int cols() const { return m; }
    int size() const { return static_cast<int>(data.size()); }

    float maxCol(int col) const
    {
        float maxVal = std::numeric_limits<float>::lowest();
        for (int i = 0; i < n; ++i)
            if (operator()(i, col) > maxVal)
                maxVal = operator()(i, col);
        return maxVal;
    }

    int maxColIdx(int col) const
    {
        int idx = 0;
        float maxVal = std::numeric_limits<float>::lowest();
        for (int i = 0; i < n; ++i)
        {
            if (operator()(i, col) > maxVal)
            {
                maxVal = operator()(i, col);
                idx = i;
            }
        }
        return idx;
    }

    float maxRow(int row) const
    {
        float maxVal = std::numeric_limits<float>::lowest();
        for (int j = 0; j < m; ++j)
            if (operator()(row, j) > maxVal)
                maxVal = operator()(row, j);
        return maxVal;
    }

    Tensor2D sliceCols(int start, int end) const
    {
        if (start < 0 || end > m || start >= end)
            throw std::out_of_range("Invalid column slice range");
        Tensor2D result(n, end - start);
        for (int i = 0; i < n; ++i)
            for (int j = start; j < end; ++j)
                result(i, j - start) = operator()(i, j);
        return result;
    }

    // i-k-j loop order for row-major cache friendliness
    Tensor2D dot(const Tensor2D &other) const
    {
        if (m != other.n)
            throw std::invalid_argument("Incompatible dimensions for multiplication");
        Tensor2D result(n, other.m);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < m; ++k)
                for (int j = 0; j < other.m; ++j)
                    result(i, j) += operator()(i, k) * other(k, j);
        return result;
    }

    Tensor2D sumColumn() const
    {
        Tensor2D result(n, 1);
        for (int i = 0; i < n; ++i)
        {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j)
                sum += operator()(i, j);
            result(i, 0) = sum;
        }
        return result;
    }

    Tensor2D transpose() const
    {
        Tensor2D result(m, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                result(j, i) = operator()(i, j);
        return result;
    }

    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    void display() const
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
                std::cout << operator()(i, j) << " ";
            std::cout << std::endl;
        }
    }
};
