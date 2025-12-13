#include <pybind11/pybind11.h>
#include "NNmain.cpp"
#include "NeuralNet.cpp"

namespace py = pybind11;

PYBIND11_MODULE(neuralnet, m) {
    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<Tensor2D,Tensor2D,Tensor2D,Tensor2D>())
        .def("get_bias1", &NeuralNet::getBias1)
        .def("get_bias2", &NeuralNet::getBias2)
        .def("get_weight1", &NeuralNet::getWeight1)
        .def("get_weight2", &NeuralNet::getWeight2);
}
