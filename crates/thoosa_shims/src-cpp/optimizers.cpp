#include "optimizers.h"


namespace {

template <typename Optimizer>
Optimizer& checked_optimizer(
    torch::optim::Optimizer* opt,
    const char* optimizer_name) {
  TORCH_CHECK(opt != nullptr, "null optimizer");

  auto* typed_optimizer = dynamic_cast<Optimizer*>(opt);
  TORCH_CHECK(typed_optimizer != nullptr, "optimizer is not torch::optim::", optimizer_name);
  return *typed_optimizer;
}

template <typename Options>
const Options& checked_options(
    const torch::optim::OptimizerOptions& options,
    const char* optimizer_name) {
  auto* typed_options = dynamic_cast<const Options*>(&options);
  TORCH_CHECK(typed_options != nullptr, "optimizer parameter group does not contain ", optimizer_name, " options");
  return *typed_options;
}

template <typename Options>
Options& checked_options(
    torch::optim::OptimizerOptions& options,
    const char* optimizer_name) {
  auto* typed_options = dynamic_cast<Options*>(&options);
  TORCH_CHECK(typed_options != nullptr, "optimizer parameter group does not contain ", optimizer_name, " options");
  return *typed_options;
}

template <typename Options>
void export_adam_like_options(
    const Options& src,
    tho_adam_options* dst,
    const char* optimizer_name) {
  TORCH_CHECK(dst != nullptr, "null ", optimizer_name, " options output");

  const auto [beta1, beta2] = src.betas();
  dst->lr = src.lr();
  dst->beta1 = beta1;
  dst->beta2 = beta2;
  dst->eps = src.eps();
  dst->weight_decay = src.weight_decay();
  dst->amsgrad = src.amsgrad() ? 1 : 0;
}

void validate_adam_like_options(
    const tho_adam_options& options,
    const char* optimizer_name) {
  TORCH_CHECK(options.lr >= 0.0, "invalid ", optimizer_name, " learning rate: ", options.lr);
  TORCH_CHECK(options.beta1 >= 0.0 && options.beta1 < 1.0, "invalid ", optimizer_name, " beta1: ", options.beta1);
  TORCH_CHECK(options.beta2 >= 0.0 && options.beta2 < 1.0, "invalid ", optimizer_name, " beta2: ", options.beta2);
  TORCH_CHECK(options.eps >= 0.0, "invalid ", optimizer_name, " epsilon: ", options.eps);
  TORCH_CHECK(options.weight_decay >= 0.0, "invalid ", optimizer_name, " weight decay: ", options.weight_decay);
}

template <typename Options>
void import_adam_like_options(
    const tho_adam_options& src,
    Options& dst,
    const char* optimizer_name) {
  validate_adam_like_options(src, optimizer_name);
  dst.lr(src.lr);
  dst.betas(std::make_tuple(src.beta1, src.beta2));
  dst.eps(src.eps);
  dst.weight_decay(src.weight_decay);
  dst.amsgrad(src.amsgrad != 0);
}

// Shape, scalar type, and device invariants apply to tensor state from every optimizer.
void validate_optimizer_state_tensor(
    const torch::Tensor& state_tensor,
    const torch::Tensor& parameter,
    const char* optimizer_name,
    const char* state_name) {
  TORCH_CHECK(state_tensor.defined(), optimizer_name, " ", state_name, " is undefined");
  TORCH_CHECK(state_tensor.sizes().vec() == parameter.sizes().vec(), optimizer_name, " ", state_name, " shape mismatch: got ", state_tensor.sizes(), ", expected ", parameter.sizes());
  TORCH_CHECK(state_tensor.scalar_type() == parameter.scalar_type(), optimizer_name, " ", state_name, " scalar type mismatch");
  TORCH_CHECK(state_tensor.device() == parameter.device(), optimizer_name, " ", state_name, " device mismatch: got ", state_tensor.device(), ", expected ", parameter.device());
}

// Adam and AdamW expose the same public state shape; templates keep their distinct LibTorch runtime types checked while sharing traversal and validation.
template <typename Optimizer>
ulong adam_like_group_count(
    torch::optim::Optimizer* opt,
    const char* optimizer_name) {
  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  return optimizer.param_groups().size();
}

template <typename Optimizer, typename Options>
void adam_like_get_defaults(
    torch::optim::Optimizer* opt,
    tho_adam_options* out,
    const char* optimizer_name) {
  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  const auto& defaults = checked_options<Options>(optimizer.defaults(), optimizer_name);
  export_adam_like_options(defaults, out, optimizer_name);
}

template <typename Optimizer, typename Options>
void adam_like_get_group(
    torch::optim::Optimizer* opt,
    size_t group_index,
    tho_adam_group* out,
    const char* optimizer_name) {
  TORCH_CHECK(out != nullptr, "null ", optimizer_name, " group output");
  *out = tho_adam_group{};

  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  const auto& group = optimizer.param_groups().at(group_index);
  const auto& options = checked_options<Options>(group.options(), optimizer_name);
  export_adam_like_options(options, &out->options, optimizer_name);
  out->parameter_count = group.params().size();
}

template <typename Optimizer, typename ParamState>
void adam_like_get_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    tho_adam_param_state* out,
    const char* optimizer_name) {
  TORCH_CHECK(out != nullptr, "null ", optimizer_name, " parameter-state output");
  *out = tho_adam_param_state{};

  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  const auto& group = optimizer.param_groups().at(group_index);
  const auto& parameter = group.params().at(parameter_index);
  const auto state_it = optimizer.state().find(parameter.unsafeGetTensorImpl());

  // Both optimizers initialize parameter state lazily, so an untouched parameter legitimately has no entry.
  if (state_it == optimizer.state().end()) {
    out->has_state = 0;
    return;
  }

  auto* state = dynamic_cast<ParamState*>(state_it->second.get());
  TORCH_CHECK(state != nullptr, "invalid ", optimizer_name, " state type");

  auto exp_avg = std::make_unique<at::Tensor>(state->exp_avg());
  auto exp_avg_sq = std::make_unique<at::Tensor>(state->exp_avg_sq());
  std::unique_ptr<at::Tensor> max_exp_avg_sq;
  if (state->max_exp_avg_sq().defined()) {
    max_exp_avg_sq = std::make_unique<at::Tensor>(state->max_exp_avg_sq());
  }

  out->step = state->step();
  out->has_state = 1;
  out->exp_avg = exp_avg.release();
  out->exp_avg_sq = exp_avg_sq.release();
  out->max_exp_avg_sq = max_exp_avg_sq.release();
}

template <typename Optimizer, typename Options>
void adam_like_set_defaults(
    torch::optim::Optimizer* opt,
    const tho_adam_options* options,
    const char* optimizer_name) {
  TORCH_CHECK(options != nullptr, "null ", optimizer_name, " defaults");
  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  auto& defaults = checked_options<Options>(optimizer.defaults(), optimizer_name);
  import_adam_like_options(*options, defaults, optimizer_name);
}

template <typename Optimizer, typename Options>
void adam_like_set_group_options(
    torch::optim::Optimizer* opt,
    size_t group_index,
    const tho_adam_options* options,
    const char* optimizer_name) {
  TORCH_CHECK(options != nullptr, "null ", optimizer_name, " group options");
  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  auto& group = optimizer.param_groups().at(group_index);
  auto& group_options = checked_options<Options>(group.options(), optimizer_name);
  import_adam_like_options(*options, group_options, optimizer_name);
}

template <typename Optimizer, typename ParamState>
void adam_like_set_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    int64_t step,
    at::Tensor* exp_avg,
    at::Tensor* exp_avg_sq,
    at::Tensor* max_exp_avg_sq,
    const char* optimizer_name) {
  auto& optimizer = checked_optimizer<Optimizer>(opt, optimizer_name);
  auto& group = optimizer.param_groups().at(group_index);
  auto& parameter = group.params().at(parameter_index);
  void* parameter_key = parameter.unsafeGetTensorImpl();

  if (has_state == 0) {
    optimizer.state().erase(parameter_key);
    return;
  }

  TORCH_CHECK(step >= 0, "negative ", optimizer_name, " step: ", step);
  TORCH_CHECK(exp_avg != nullptr && exp_avg->defined(), optimizer_name, " exp_avg is undefined");
  TORCH_CHECK(exp_avg_sq != nullptr && exp_avg_sq->defined(), optimizer_name, " exp_avg_sq is undefined");
  validate_optimizer_state_tensor(*exp_avg, parameter, optimizer_name, "exp_avg");
  validate_optimizer_state_tensor(*exp_avg_sq, parameter, optimizer_name, "exp_avg_sq");
  if (max_exp_avg_sq != nullptr && max_exp_avg_sq->defined()) {
    validate_optimizer_state_tensor(*max_exp_avg_sq, parameter, optimizer_name, "max_exp_avg_sq");
  }

  auto state = std::make_unique<ParamState>();
  state->step(step);
  state->exp_avg(*exp_avg);
  state->exp_avg_sq(*exp_avg_sq);
  if (max_exp_avg_sq != nullptr && max_exp_avg_sq->defined()) {
    state->max_exp_avg_sq(*max_exp_avg_sq);
  }

  // Tensor assignments retain intrusive handles, so Rust can release its snapshot tensors after this call.
  optimizer.state()[parameter_key] = std::move(state);
}

void export_sgd_options(
    const torch::optim::SGDOptions& src,
    tho_sgd_options* dst) {
  TORCH_CHECK(dst != nullptr, "null SGD options output");
  dst->lr = src.lr();
  dst->momentum = src.momentum();
  dst->dampening = src.dampening();
  dst->weight_decay = src.weight_decay();
  dst->nesterov = src.nesterov() ? 1 : 0;
}

void validate_sgd_options(const tho_sgd_options& options) {
  TORCH_CHECK(options.lr >= 0.0, "invalid SGD learning rate: ", options.lr);
  TORCH_CHECK(options.momentum >= 0.0, "invalid SGD momentum: ", options.momentum);
  TORCH_CHECK(options.weight_decay >= 0.0, "invalid SGD weight decay: ", options.weight_decay);
  TORCH_CHECK(options.nesterov == 0 || (options.momentum > 0.0 && options.dampening == 0.0), "SGD Nesterov momentum requires positive momentum and zero dampening");
}

void import_sgd_options(
    const tho_sgd_options& src,
    torch::optim::SGDOptions& dst) {
  validate_sgd_options(src);
  dst.lr(src.lr);
  dst.momentum(src.momentum);
  dst.dampening(src.dampening);
  dst.weight_decay(src.weight_decay);
  dst.nesterov(src.nesterov != 0);
}

} // namespace


extern "C" ulong tho_adam_group_count(torch::optim::Optimizer* opt) {
  return adam_like_group_count<torch::optim::Adam>(opt, "Adam");
}

extern "C" void tho_adam_get_defaults(torch::optim::Optimizer* opt, tho_adam_options* out) {
  adam_like_get_defaults<torch::optim::Adam, torch::optim::AdamOptions>(opt, out, "Adam");
}

extern "C" void tho_adam_get_group(torch::optim::Optimizer* opt, size_t group_index, tho_adam_group* out) {
  adam_like_get_group<torch::optim::Adam, torch::optim::AdamOptions>(opt, group_index, out, "Adam");
}

extern "C" void tho_adam_get_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, tho_adam_param_state* out) {
  adam_like_get_param_state<torch::optim::Adam, torch::optim::AdamParamState>(opt, group_index, parameter_index, out, "Adam");
}

extern "C" void tho_adam_set_defaults(torch::optim::Optimizer* opt, const tho_adam_options* options) {
  adam_like_set_defaults<torch::optim::Adam, torch::optim::AdamOptions>(opt, options, "Adam");
}

extern "C" void tho_adam_set_group_options(torch::optim::Optimizer* opt, size_t group_index, const tho_adam_options* options) {
  adam_like_set_group_options<torch::optim::Adam, torch::optim::AdamOptions>(opt, group_index, options, "Adam");
}

extern "C" void tho_adam_set_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, int32_t has_state, int64_t step, at::Tensor* exp_avg, at::Tensor* exp_avg_sq, at::Tensor* max_exp_avg_sq) {
  adam_like_set_param_state<torch::optim::Adam, torch::optim::AdamParamState>(opt, group_index, parameter_index, has_state, step, exp_avg, exp_avg_sq, max_exp_avg_sq, "Adam");
}


extern "C" ulong tho_adam_w_group_count(torch::optim::Optimizer* opt) {
  return adam_like_group_count<torch::optim::AdamW>(opt, "AdamW");
}

extern "C" void tho_adam_w_get_defaults(torch::optim::Optimizer* opt, tho_adam_options* out) {
  adam_like_get_defaults<torch::optim::AdamW, torch::optim::AdamWOptions>(opt, out, "AdamW");
}

extern "C" void tho_adam_w_get_group(torch::optim::Optimizer* opt, size_t group_index, tho_adam_group* out) {
  adam_like_get_group<torch::optim::AdamW, torch::optim::AdamWOptions>(opt, group_index, out, "AdamW");
}

extern "C" void tho_adam_w_get_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, tho_adam_param_state* out) {
  adam_like_get_param_state<torch::optim::AdamW, torch::optim::AdamWParamState>(opt, group_index, parameter_index, out, "AdamW");
}

extern "C" void tho_adam_w_set_defaults(torch::optim::Optimizer* opt, const tho_adam_options* options) {
  adam_like_set_defaults<torch::optim::AdamW, torch::optim::AdamWOptions>(opt, options, "AdamW");
}

extern "C" void tho_adam_w_set_group_options(torch::optim::Optimizer* opt, size_t group_index, const tho_adam_options* options) {
  adam_like_set_group_options<torch::optim::AdamW, torch::optim::AdamWOptions>(opt, group_index, options, "AdamW");
}

extern "C" void tho_adam_w_set_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, int32_t has_state, int64_t step, at::Tensor* exp_avg, at::Tensor* exp_avg_sq, at::Tensor* max_exp_avg_sq) {
  adam_like_set_param_state<torch::optim::AdamW, torch::optim::AdamWParamState>(opt, group_index, parameter_index, has_state, step, exp_avg, exp_avg_sq, max_exp_avg_sq, "AdamW");
}


extern "C" ulong tho_sgd_group_count(torch::optim::Optimizer* opt) {
  return checked_optimizer<torch::optim::SGD>(opt, "SGD").param_groups().size();
}

extern "C" void tho_sgd_get_defaults(torch::optim::Optimizer* opt, tho_sgd_options* out) {
  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  const auto& defaults = checked_options<torch::optim::SGDOptions>(sgd.defaults(), "SGD");
  export_sgd_options(defaults, out);
}

extern "C" void tho_sgd_get_group(torch::optim::Optimizer* opt, size_t group_index, tho_sgd_group* out) {
  TORCH_CHECK(out != nullptr, "null SGD group output");
  *out = tho_sgd_group{};

  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  const auto& group = sgd.param_groups().at(group_index);
  const auto& options = checked_options<torch::optim::SGDOptions>(group.options(), "SGD");
  export_sgd_options(options, &out->options);
  out->parameter_count = group.params().size();
}

extern "C" void tho_sgd_get_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, tho_sgd_param_state* out) {
  TORCH_CHECK(out != nullptr, "null SGD parameter-state output");
  *out = tho_sgd_param_state{};

  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  const auto& group = sgd.param_groups().at(group_index);
  const auto& parameter = group.params().at(parameter_index);
  const auto state_it = sgd.state().find(parameter.unsafeGetTensorImpl());

  // SGD without momentum legitimately has no per-parameter state.
  if (state_it == sgd.state().end()) {
    out->has_state = 0;
    return;
  }

  auto* state = dynamic_cast<torch::optim::SGDParamState*>(state_it->second.get());
  TORCH_CHECK(state != nullptr, "invalid SGD state type");
  TORCH_CHECK(state->momentum_buffer().defined(), "SGD momentum buffer is undefined");

  auto momentum_buffer = std::make_unique<at::Tensor>(state->momentum_buffer());
  out->has_state = 1;
  out->momentum_buffer = momentum_buffer.release();
}

extern "C" void tho_sgd_set_defaults(torch::optim::Optimizer* opt, const tho_sgd_options* options) {
  TORCH_CHECK(options != nullptr, "null SGD defaults");
  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  auto& defaults = checked_options<torch::optim::SGDOptions>(sgd.defaults(), "SGD");
  import_sgd_options(*options, defaults);
}

extern "C" void tho_sgd_set_group_options(torch::optim::Optimizer* opt, size_t group_index, const tho_sgd_options* options) {
  TORCH_CHECK(options != nullptr, "null SGD group options");
  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  auto& group = sgd.param_groups().at(group_index);
  auto& group_options = checked_options<torch::optim::SGDOptions>(group.options(), "SGD");
  import_sgd_options(*options, group_options);
}

extern "C" void tho_sgd_set_param_state(torch::optim::Optimizer* opt, size_t group_index, size_t parameter_index, int32_t has_state, at::Tensor* momentum_buffer) {
  auto& sgd = checked_optimizer<torch::optim::SGD>(opt, "SGD");
  auto& group = sgd.param_groups().at(group_index);
  auto& parameter = group.params().at(parameter_index);
  void* parameter_key = parameter.unsafeGetTensorImpl();

  if (has_state == 0) {
    sgd.state().erase(parameter_key);
    return;
  }

  TORCH_CHECK(momentum_buffer != nullptr && momentum_buffer->defined(), "SGD momentum buffer is undefined");
  validate_optimizer_state_tensor(*momentum_buffer, parameter, "SGD", "momentum_buffer");

  auto state = std::make_unique<torch::optim::SGDParamState>();
  state->momentum_buffer(*momentum_buffer);

  // The state retains its own intrusive tensor handle after Rust releases the snapshot tensor.
  sgd.state()[parameter_key] = std::move(state);
}
