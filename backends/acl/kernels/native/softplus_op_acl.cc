#include <memory>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_softplus_backward.h>
#include "softplus_op_acl.h"

namespace jittor {
namespace {
using Scalar = std::unique_ptr<aclScalar, decltype(&aclDestroyScalar)>;

Scalar float_scalar(float value) {
    return Scalar(aclCreateScalar(&value, ACL_FLOAT), aclDestroyScalar);
}
}

SoftplusOpRunner::SoftplusOpRunner() : BaseOpRunner("Softplus") {}

void SoftplusOpRunner::executeOp(AclOpRegistry::const_iterator &) {
    auto beta = float_scalar(1);
    auto threshold = float_scalar(20);
    ret = aclnnSoftplusGetWorkspaceSize(inputTensors[0], beta.get(), threshold.get(),
        outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnSoftplus, true);
}

SoftplusBackwardOpRunner::SoftplusBackwardOpRunner()
    : BaseOpRunner("SoftplusBackward") {}

void SoftplusBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &) {
    auto beta = float_scalar(1);
    auto threshold = float_scalar(20);
    ret = aclnnSoftplusBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1],
        beta.get(), threshold.get(), outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnSoftplusBackward, true);
}
}
