#pragma once
#include "base_op.h"

namespace jittor {
class SoftplusOpRunner : public BaseOpRunner {
protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;

public:
    SoftplusOpRunner();
};

class SoftplusBackwardOpRunner : public BaseOpRunner {
protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;

public:
    SoftplusBackwardOpRunner();
};
}
