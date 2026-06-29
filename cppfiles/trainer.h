#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include "nn_model.h"
#include "optimizer.h"
#include "tensor.h"

// Permute les colonnes de pixels et labels avec le même ordre aléatoire
inline void shuffleData(Tensor2D &pixels, Tensor2D &labels, std::mt19937 &rng)
{
    int n = pixels.cols();
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    Tensor2D newPix(pixels.rows(), n), newLbl(labels.rows(), n);
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < pixels.rows(); ++r)
            newPix(r, i) = pixels(r, perm[i]);
        for (int r = 0; r < labels.rows(); ++r)
            newLbl(r, i) = labels(r, perm[i]);
    }
    pixels = std::move(newPix);
    labels = std::move(newLbl);
}

// Moyenne les gradients de plusieurs threads
static inline Gradients averageGradients(const std::vector<Gradients> &gs, int count)
{
    Gradients avg = gs[0];
    for (int t = 1; t < count; ++t) {
        avg.dw1 = avg.dw1 + gs[t].dw1;
        avg.db1 = avg.db1 + gs[t].db1;
        avg.dw2 = avg.dw2 + gs[t].dw2;
        avg.db2 = avg.db2 + gs[t].db2;
    }
    float inv = 1.0f / count;
    avg.dw1 = avg.dw1 * inv;
    avg.db1 = avg.db1 * inv;
    avg.dw2 = avg.dw2 * inv;
    avg.db2 = avg.db2 * inv;
    return avg;
}

// Mini-batch SGD avec momentum, decay et reporting sur le dev set.
// Chaque thread calcule les gradients sur sa tranche ; on moyenne avant l'update.
inline NetworkParams train(
    Tensor2D trainX, Tensor2D trainY,
    const Tensor2D &devX, const Tensor2D &devY,
    SGDMomentum &opt,
    int epochs, int batchSize, int numThreads)
{
    NetworkParams params = initParameters();

    int N = trainX.cols();
    std::vector<Gradients> threadGrads(numThreads);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        shuffleData(trainX, trainY, rng);

        for (int bs = 0; bs < N; bs += batchSize)
        {
            int be         = std::min(bs + batchSize, N);
            int actual     = be - bs;
            int effThreads = std::min(numThreads, actual);
            int sub        = actual / effThreads;

            auto process = [&](int t, int absStart, int absEnd)
            {
                Tensor2D bX  = trainX.sliceCols(absStart, absEnd);
                Tensor2D bY  = trainY.sliceCols(absStart, absEnd);
                Tensor2D bOH = oneHot(bY);
                ForwardCache cache = forwardPass(bX, params);
                threadGrads[t]     = backwardPass(bX, cache, bOH, params);
            };

            std::vector<std::thread> threads;
            threads.reserve(effThreads);
            for (int t = 0; t < effThreads; ++t) {
                int ts = bs + t * sub;
                int te = (t == effThreads - 1) ? be : bs + (t + 1) * sub;
                threads.emplace_back(process, t, ts, te);
            }
            for (auto &th : threads) th.join();

            opt.step(params, averageGradients(threadGrads, effThreads));
        }

        opt.epochEnd();

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            ForwardCache devCache = forwardPass(devX, params);
            float devLoss = crossEntropyLoss(devCache.softmax, oneHot(devY));
            float devAcc  = accuracy(devCache, devY);
            std::cout << "Epoch " << std::setw(3) << epoch
                      << "  |  loss: "    << std::fixed      << std::setprecision(4) << devLoss
                      << "  |  dev_acc: " << std::fixed      << std::setprecision(2) << devAcc  << "%"
                      << "  |  lr: "      << std::scientific << std::setprecision(3) << opt.lr() << "\n";
        }
    }

    return params;
}
