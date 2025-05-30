#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <initializer_list>

using namespace std;


class Tensor2D {
private:
    int n, m;
    vector<float> data;

    int computeOffset(int i, int j) const {
        if (i >= n || j >= m)
            throw out_of_range("Index out of bounds");
        return i * m + j;
    }


public:
    Tensor2D() = default;
    // Constructor to initialize a 2D tensor with given dimensions
    Tensor2D(int n, int m)
        : n(n), m(m), data(n * m, 0.0f) {}

    float& operator()(int i, int j) {
        return data[computeOffset(i, j)];
    }

    float operator()(int i, int j) const {
        return data[computeOffset(i, j)];
    }

    void copy(const Tensor2D& other) {
    if (this != &other) {
        n = other.n;
        m = other.m;
        data = other.data;
    };
}

    int rows() const { return n; }
    int cols() const { return m; }
    int size() const { return data.size(); }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    void display() const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cout << operator()(i, j) << " ";
            }
            cout << endl;
        }
    }

    void print() const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cout << operator()(i, j) << " ";
            }
            cout << endl;
        }
    }
};

// Example usage
/*
int main() {
    Tensor2D t(2, 3);
    t.fill(1.5f);
    t(1, 2) = 5.0f;
    t.print();
    cout << "Shape: " << t.rows() << " " << t.cols() << endl;
    cout << "Element at (1,2): " << t(1, 2) << endl;
    return 0;
}
*/

