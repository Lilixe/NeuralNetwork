#include <iostream>
#include "tensor.h"
#include "config.h"
#include "data_loader.h"
#include "nn_model.h"
#include "optimizer.h"
#include "trainer.h"
#include "io.h"

int main()
{
    // ── Hyperparameters ───────────────────────────────────────────────────────
    const int   epochs     = 50;
    const float lr         = 0.05f;
    const float lrDecay    = 0.99f;
    const float beta       = 0.9f;
    const int   batchSize  = 512;
    const int   numThreads = 4;

    // ── Data ──────────────────────────────────────────────────────────────────
    std::cout << "Loading data...\n";
    auto testData  = loadMNIST("../mnist_test.csv");
    auto trainData = loadMNIST("../mnist_train.csv");

    Tensor2D testLabels, testPixels;
    std::tie(testLabels, testPixels) = toMatrix(testData);

    DataSet ds = initData(trainData);

    // ── Training ──────────────────────────────────────────────────────────────
    std::cout << "\n==============================\n";
    SGDMomentum opt(lr, beta, lrDecay);
    NetworkParams params = train(
        ds.trainPixels, ds.trainLabels,
        ds.devPixels,   ds.devLabels,
        opt, epochs, batchSize, numThreads);

    saveParameters(params, "nn_parameters.txt");

    // ── Evaluation ────────────────────────────────────────────────────────────
    ForwardCache testCache = forwardPass(testPixels, params);
    std::cout << "\n==============================\n";
    std::cout << "Test accuracy: " << accuracy(testCache, testLabels) << "%\n";
    std::cout << "==============================\n";

    return 0;
}
