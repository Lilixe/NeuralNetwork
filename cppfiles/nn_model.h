#pragma once
#include <cmath>
#include <random>
#include "tensor.h"
#include "config.h"

// ── Structs ──────────────────────────────────────────────────────────────────

struct NetworkParams {
    Tensor2D w1, b1, w2, b2;
};

struct ForwardCache {
    Tensor2D l1, l1relu, l2, softmax;
};

struct Gradients {
    Tensor2D dw1, db1, dw2, db2;
    float loss = 0.0f;
};

// ── Initialization ────────────────────────────────────────────────────────────

inline NetworkParams initParameters()
{
    NetworkParams p;
    p.w1 = Tensor2D(HIDDEN_SIZE, INPUT_SIZE);
    p.b1 = Tensor2D(HIDDEN_SIZE, 1);          // zero-initialized
    p.w2 = Tensor2D(OUTPUT_SIZE, HIDDEN_SIZE);
    p.b2 = Tensor2D(OUTPUT_SIZE, 1);          // zero-initialized

    std::random_device rd;
    std::mt19937 gen(rd());
    // He init: N(0, sqrt(2/fan_in))
    std::normal_distribution<float> dw1(0.0f, std::sqrtf(2.0f / INPUT_SIZE));
    std::normal_distribution<float> dw2(0.0f, std::sqrtf(2.0f / HIDDEN_SIZE));

    for (int i = 0; i < HIDDEN_SIZE; ++i)
        for (int j = 0; j < INPUT_SIZE; ++j)
            p.w1(i, j) = dw1(gen);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            p.w2(i, j) = dw2(gen);

    return p;
}

// ── Activation helpers (internal) ────────────────────────────────────────────

static inline Tensor2D relu(const Tensor2D &t)
{
    Tensor2D result(t.rows(), t.cols());
    for (int i = 0; i < t.rows(); ++i)
        for (int j = 0; j < t.cols(); ++j)
            result(i, j) = (t(i, j) < 0.0f) ? 0.0f : t(i, j);
    return result;
}

static inline Tensor2D softmax(const Tensor2D &t)
{
    Tensor2D result(t.rows(), t.cols());
    for (int j = 0; j < t.cols(); ++j)
    {
        float maxVal = t.maxCol(j);
        float sum    = 0.0f;
        for (int i = 0; i < t.rows(); ++i) {
            result(i, j) = std::expf(t(i, j) - maxVal);
            sum += result(i, j);
        }
        for (int i = 0; i < t.rows(); ++i)
            result(i, j) /= sum;
    }
    return result;
}

static inline Tensor2D derivativeRelu(const Tensor2D &grad, const Tensor2D &preRelu)
{
    Tensor2D result(grad.rows(), grad.cols());
    for (int i = 0; i < grad.rows(); ++i)
        for (int j = 0; j < grad.cols(); ++j)
            result(i, j) = (preRelu(i, j) > 0.0f) ? grad(i, j) : 0.0f;
    return result;
}

// ── Public math operations ────────────────────────────────────────────────────

inline Tensor2D oneHot(const Tensor2D &labels)
{
    Tensor2D result(labels.cols(), OUTPUT_SIZE);
    for (int i = 0; i < labels.cols(); ++i)
        result(i, static_cast<int>(labels(0, i))) = 1.0f;
    return result.transpose();
}

inline float crossEntropyLoss(const Tensor2D &preds, const Tensor2D &oneHotY)
{
    float loss = 0.0f;
    int m = preds.cols();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < preds.rows(); ++j)
            loss += oneHotY(j, i) * std::logf(preds(j, i) + 1e-8f);
    return -loss / m;
}

// ── Forward / Backward ───────────────────────────────────────────────────────

inline ForwardCache forwardPass(const Tensor2D &X, const NetworkParams &p)
{
    ForwardCache c;
    c.l1      = p.w1.dot(X) + p.b1;
    c.l1relu  = relu(c.l1);
    c.l2      = p.w2.dot(c.l1relu) + p.b2;
    c.softmax = softmax(c.l2);
    return c;
}

inline Gradients backwardPass(const Tensor2D &X, const ForwardCache &c,
                               const Tensor2D &oneHotY, const NetworkParams &p)
{
    int m = c.softmax.cols();

    Tensor2D dl2 = c.softmax - oneHotY;
    Tensor2D dw2 = dl2.dot(c.l1relu.transpose()) * (1.0f / m);
    Tensor2D db2 = dl2.sumColumn() * (1.0f / m);

    Tensor2D dl1 = derivativeRelu(p.w2.transpose().dot(dl2), c.l1);
    Tensor2D dw1 = dl1.dot(X.transpose()) * (1.0f / m);
    Tensor2D db1 = dl1.sumColumn() * (1.0f / m);

    Gradients g;
    g.dw1 = dw1;  g.db1 = db1;
    g.dw2 = dw2;  g.db2 = db2;
    g.loss = crossEntropyLoss(c.softmax, oneHotY);
    return g;
}

// ── Metrics ──────────────────────────────────────────────────────────────────

inline float accuracy(const ForwardCache &c, const Tensor2D &labels)
{
    float correct = 0.0f;
    for (int i = 0; i < c.softmax.cols(); ++i)
        if (c.softmax.maxColIdx(i) == static_cast<int>(labels(0, i)))
            ++correct;
    return (correct / c.softmax.cols()) * 100.0f;
}
