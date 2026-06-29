#pragma once
#include "tensor.h"
#include "nn_model.h"

// SGD with heavy-ball momentum and optional exponential learning rate decay.
// Velocity tensors are lazily initialized on the first call to step().
class SGDMomentum {
public:
    SGDMomentum(float lr, float beta, float decay)
        : lr_(lr), beta_(beta), decay_(decay) {}

    // v = beta*v + grad  ;  param = param - lr*v
    void step(NetworkParams &p, const Gradients &g)
    {
        initVelocityIfNeeded(g);

        vel_.dw1 = vel_.dw1 * beta_ + g.dw1;
        vel_.db1 = vel_.db1 * beta_ + g.db1;
        vel_.dw2 = vel_.dw2 * beta_ + g.dw2;
        vel_.db2 = vel_.db2 * beta_ + g.db2;

        p.w1 = p.w1 - vel_.dw1 * lr_;
        p.b1 = p.b1 - vel_.db1 * lr_;
        p.w2 = p.w2 - vel_.dw2 * lr_;
        p.b2 = p.b2 - vel_.db2 * lr_;
    }

    // Call at end of each epoch to apply learning rate decay
    void epochEnd() { lr_ *= decay_; }

    float lr() const { return lr_; }

private:
    float lr_, beta_, decay_;
    Gradients vel_;

    void initVelocityIfNeeded(const Gradients &g)
    {
        if (vel_.dw1.rows() != 0) return;
        vel_.dw1 = Tensor2D(g.dw1.rows(), g.dw1.cols());
        vel_.db1 = Tensor2D(g.db1.rows(), g.db1.cols());
        vel_.dw2 = Tensor2D(g.dw2.rows(), g.dw2.cols());
        vel_.db2 = Tensor2D(g.db2.rows(), g.db2.cols());
    }
};
