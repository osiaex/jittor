#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        // Every CANN binary broadcasts its operands, so a one-element operand
        // may be handed over as a one-element tensor.
        bool collapsesScalarInputs() const override { return true; }

        void executeOp(AclOpRegistry::const_iterator &it) override;
    };

    class PowScalarOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        double exponent = 0;
        PowScalarOpRunner() : BaseOpRunner("PowScalar") {}
    };
}
