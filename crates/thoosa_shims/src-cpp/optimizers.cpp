
#include "optimizers.h"

extern "C" void tho_snapshot_adam_states(
    // torch::optim::Optimizer& base
) {
    tho_log(CppLogLevel::Debug, "snapshot_adam_states");
}
