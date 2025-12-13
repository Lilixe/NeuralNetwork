#include "tensor.cpp"

class NeuralNet {
public: 
    NeuralNet(Tensor2D w1, Tensor2D w2, Tensor2D b1, Tensor2D b2);
    Tensor2D getWeight1() {
        return w1;
    }
    Tensor2D getWeight2() {
        return w2;
    }
    Tensor2D getBias1() {
        return b1;
    }
    Tensor2D getBias2() {
        return b2;
    }

private:
    Tensor2D w1;
    Tensor2D w2;
    Tensor2D b1;
    Tensor2D b2;
};