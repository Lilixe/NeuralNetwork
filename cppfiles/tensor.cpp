#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <initializer_list>
#include <algorithm>


using namespace std;


class Tensor2D
{
private:
    int n, m;
    vector<float> data;

    int computeOffset(int i, int j) const
    {
        if (i >= n || j >= m)
            throw out_of_range("Index out of bounds");
        return i * m + j;
    }

public:
    // Default constructor initializes a 2D tensor with zero dimensions
    Tensor2D() = default;

    // Constructor to initialize a 2D tensor with an other tensor
    Tensor2D(const Tensor2D &other)
        : n(other.n), m(other.m), data(other.data) {}

    // Constructor to initialize a 2D tensor with given dimensions
    Tensor2D(int n, int m)
        : n(n), m(m), data(n * m, 0.0f) {}

    // Overloaded operator to access elements in a tensor
    float &operator()(int i, int j)
    {
        return data[computeOffset(i, j)];
    }

    // Overloaded operator to access elements in a const tensor
    float operator()(int i, int j) const
    {
        return data[computeOffset(i, j)];
    }

    // Overloaded operator for addition of a tensor and a column vector or a 2D tensor of same shape
    Tensor2D operator+(const Tensor2D &other) const
    {
        // Add another 2D tensor of the same shape
        if (n == other.n && m == other.m)
        {
            Tensor2D result(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    result(i, j) = operator()(i, j) + other(i, j);
            return result;
        }
        // Add a column vector (other.m == 1, other.n == n)
        if (other.m == 1 && other.n == n)
        {
            Tensor2D result(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    result(i, j) = operator()(i, j) + other(i, 0);
            return result;
        }
        throw invalid_argument("Tensors must have the same dimensions or other must be a column vector with matching rows for addition");
    }

    // Overloaded operator for subtraction
    Tensor2D operator-(const Tensor2D &other) const
    {
        if (n != other.n || m != other.m)
            throw invalid_argument("Tensors must have the same dimensions for subtraction");

        Tensor2D result = Tensor2D(n, m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                result(i, j) = operator()(i, j) - other(i, j);
            }
        }
        return result;
    }

    // Overloaded assignment operator
    Tensor2D &operator=(const Tensor2D &other)
    {
        if (this != &other)
        {
            n = other.n;
            m = other.m;
            data = other.data;
        }
        return *this;
    }

    // function to acess size and dimensions of the tensor
    int rows() const { return n; }
    int cols() const { return m; }
    int size() const { return data.size(); }

    // Get the maximum value in the tensor's column
    float max(int col) const
    {
        float maxVal = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (operator()(i, col) > maxVal)
            {
                maxVal = operator()(i, col);
            }
        }
        return maxVal;
    }

    float sum() const
    {
        float total = 0.0f;
        for (const float &value : data)
        {
            total += value;
        }
        return total;
    }

    // Matrix multiplication
    Tensor2D dot(const Tensor2D &other) const
    {
        if (m != other.n)
            throw invalid_argument("Incompatible dimensions for multiplication");

        Tensor2D result = Tensor2D(n, other.m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < other.m; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < m; ++k)
                {
                    sum += operator()(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // function to concatenate two tensors
    Tensor2D concatenate(const Tensor2D &other) const
    {
        if (n != other.n)
            throw invalid_argument("Tensors must have the same number of rows for concatenation");

        Tensor2D result = Tensor2D(n, m + other.m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                result(i, j) = operator()(i, j);
            }
            for (int j = 0; j < other.m; ++j)
            {
                result(i, m + j) = other(i, j);
            }
        }
        return result;
    }

    // function T to transpose a tensor
    Tensor2D transpose() const
    {
        Tensor2D result = Tensor2D(m, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                result(j, i) = operator()(i, j);
            }
        }
        return result;
    }

    // Function to fill the tensor with a specific value
    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    // Function to display the tensor values
    void display() const
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                cout << operator()(i, j) << " ";
            }
            cout << endl;
        }
    }
};

/*
// Example usage
int main() {
    Tensor2D t(2, 3);
    t.fill(1.5f);
    t(1, 2) = 5.0f;
    t.display();
    cout << "Shape: " << t.rows() << " " << t.cols() << endl;
    cout << "Element at (1,2): " << t(1, 2) << endl;
    return 0;
}*/