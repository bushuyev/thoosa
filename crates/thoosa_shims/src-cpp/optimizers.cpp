
#include "optimizers.h"



namespace {

torch::optim::Adam& checked_adam(torch::optim::Optimizer* opt) {
  TORCH_CHECK(opt != nullptr, "null optimizer");

  auto* adam = dynamic_cast<torch::optim::Adam*>(opt);

  TORCH_CHECK(
      adam != nullptr,
      "optimizer is not torch::optim::Adam");

  return *adam;
}

const torch::optim::AdamOptions& checked_adam_options(
    const torch::optim::OptimizerOptions& options) {
  auto* adam_options =
      dynamic_cast<const torch::optim::AdamOptions*>(&options);

  TORCH_CHECK(
      adam_options != nullptr,
      "optimizer parameter group does not contain AdamOptions");

  return *adam_options;
}

torch::optim::AdamOptions& checked_adam_options(
    torch::optim::OptimizerOptions& options) {
  auto* adam_options =
      dynamic_cast<torch::optim::AdamOptions*>(&options);

  TORCH_CHECK(
      adam_options != nullptr,
      "optimizer parameter group does not contain AdamOptions");

  return *adam_options;
}

void export_adam_options(
    const torch::optim::AdamOptions& src,
    tho_adam_options* dst) {
  TORCH_CHECK(dst != nullptr, "null Adam options output");

  const auto [beta1, beta2] = src.betas();

  dst->lr = src.lr();
  dst->beta1 = beta1;
  dst->beta2 = beta2;
  dst->eps = src.eps();
  dst->weight_decay = src.weight_decay();
  dst->amsgrad = src.amsgrad() ? 1 : 0;
}

void validate_adam_options(
    const tho_adam_options& options) {
  TORCH_CHECK(
      options.lr >= 0.0,
      "invalid Adam learning rate: ",
      options.lr);

  TORCH_CHECK(
      options.beta1 >= 0.0 && options.beta1 < 1.0,
      "invalid Adam beta1: ",
      options.beta1);

  TORCH_CHECK(
      options.beta2 >= 0.0 && options.beta2 < 1.0,
      "invalid Adam beta2: ",
      options.beta2);

  TORCH_CHECK(
      options.eps >= 0.0,
      "invalid Adam epsilon: ",
      options.eps);

  TORCH_CHECK(
      options.weight_decay >= 0.0,
      "invalid Adam weight decay: ",
      options.weight_decay);
}

void import_adam_options(
    const tho_adam_options& src,
    torch::optim::AdamOptions& dst) {
  validate_adam_options(src);

  dst.lr(src.lr);
  dst.betas(std::make_tuple(src.beta1, src.beta2));
  dst.eps(src.eps);
  dst.weight_decay(src.weight_decay);
  dst.amsgrad(src.amsgrad != 0);
}

void validate_adam_state_tensor(
    const torch::Tensor& state_tensor,
    const torch::Tensor& parameter,
    const char* state_name) {
  TORCH_CHECK(
      state_tensor.defined(),
      state_name,
      " is undefined");

  TORCH_CHECK(
      state_tensor.sizes().vec() == parameter.sizes().vec(),
      state_name,
      " shape mismatch: got ",
      state_tensor.sizes(),
      ", expected ",
      parameter.sizes());

  TORCH_CHECK(
      state_tensor.scalar_type() == parameter.scalar_type(),
      state_name,
      " scalar type mismatch");

  TORCH_CHECK(
      state_tensor.device() == parameter.device(),
      state_name,
      " device mismatch: got ",
      state_tensor.device(),
      ", expected ",
      parameter.device());
}

} // namespace

extern "C" ulong tho_adam_group_count(torch::optim::Optimizer* opt) {

    tho_log(CppLogLevel::Debug, "tho_adam_group_count");

    auto& adam = checked_adam(opt);

    tho_log(CppLogLevel::Debug, "tho_adam_group_count checked_adam ok");

    return adam.param_groups().size();


    return 0;
}

extern "C"  void tho_adam_get_defaults( torch::optim::Optimizer* opt, tho_adam_options* out) {

    TORCH_CHECK(out != nullptr, "null Adam defaults output");

    auto& adam = checked_adam(opt);

    const auto& defaults =
        checked_adam_options(adam.defaults());

    export_adam_options(defaults, out);

}

extern "C"  void tho_adam_get_group( torch::optim::Optimizer* opt, size_t group_index, tho_adam_group* out) {

    TORCH_CHECK(out != nullptr, "null Adam group output");

    *out = tho_adam_group{};

    auto& adam = checked_adam(opt);
    const auto& group =
        adam.param_groups().at(group_index);

    const auto& options =
        checked_adam_options(group.options());

    export_adam_options(options, &out->options);
    out->parameter_count = group.params().size();

}

extern "C"  void tho_adam_get_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    tho_adam_param_state* out) {
    tho_log(CppLogLevel::Debug, "tho_adam_get_param_state");

    TORCH_CHECK(
        out != nullptr,
        "null Adam parameter-state output");

    *out = tho_adam_param_state{};

    auto& adam = checked_adam(opt);
    const auto& group =
        adam.param_groups().at(group_index);
    const auto& parameter =
        group.params().at(parameter_index);

    const auto state_it =
        adam.state().find(
            parameter.unsafeGetTensorImpl());

    // Adam initializes state lazily. A parameter that has never
    // participated in step() legitimately has no state.
    if (state_it == adam.state().end()) {
        out->has_state = 0;
        return;
    }

    auto* state =
          dynamic_cast<torch::optim::AdamParamState*>(
              state_it->second.get());

    TORCH_CHECK(
        state != nullptr,
        "invalid Adam state type");

    auto exp_avg =
        std::make_unique<torch::Tensor>(state->exp_avg());

    auto exp_avg_sq =
        std::make_unique<torch::Tensor>(state->exp_avg_sq());

    std::unique_ptr<torch::Tensor> max_exp_avg_sq;

    if (state->max_exp_avg_sq().defined()) {
        max_exp_avg_sq =
            std::make_unique<torch::Tensor>(
                state->max_exp_avg_sq());
    }

    out->step = state->step();
    out->has_state = 1;
    out->exp_avg = exp_avg.release();
    out->exp_avg_sq = exp_avg_sq.release();
    out->max_exp_avg_sq = max_exp_avg_sq.release();

}

extern "C"  void tho_adam_set_defaults( torch::optim::Optimizer* opt, const tho_adam_options* options) {

    TORCH_CHECK(options != nullptr, "null Adam defaults");

    auto& adam = checked_adam(opt);
    auto& defaults =
        checked_adam_options(adam.defaults());

    import_adam_options(*options, defaults);

}

extern "C"  void tho_adam_set_group_options(
    torch::optim::Optimizer* opt,
    size_t group_index,
    const tho_adam_options* options) {

    TORCH_CHECK(options != nullptr, "null Adam group options");

    auto& adam = checked_adam(opt);
    auto& group =
        adam.param_groups().at(group_index);

    auto& group_options =
        checked_adam_options(group.options());

    import_adam_options(*options, group_options);

}

void tho_adam_set_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    int64_t step,
    at::Tensor* exp_avg,
    at::Tensor* exp_avg_sq,
    at::Tensor* max_exp_avg_sq) {

    auto& adam = checked_adam(opt);
    auto& group =
        adam.param_groups().at(group_index);
    auto& parameter =
        group.params().at(parameter_index);

    void* parameter_key =
        parameter.unsafeGetTensorImpl();

    if (has_state == 0) {
        adam.state().erase(parameter_key);
        return;
    }

    TORCH_CHECK(
        step >= 0,
        "negative Adam step: ",
        step);

    TORCH_CHECK(
        exp_avg->defined(),
        "Adam exp_avg is undefined");

    TORCH_CHECK(
        exp_avg_sq->defined(),
        "Adam exp_avg_sq is undefined");

    validate_adam_state_tensor(
        *exp_avg,
        parameter,
        "Adam exp_avg");

    validate_adam_state_tensor(
        *exp_avg_sq,
        parameter,
        "Adam exp_avg_sq");

    if (max_exp_avg_sq != nullptr && max_exp_avg_sq->defined()) {
        validate_adam_state_tensor(
            *max_exp_avg_sq,
            parameter,
            "Adam max_exp_avg_sq");
    }

    auto state =
        std::make_unique<torch::optim::AdamParamState>();

    state->step(step);
    state->exp_avg(*exp_avg);
    state->exp_avg_sq(*exp_avg_sq);

    if (max_exp_avg_sq != nullptr && max_exp_avg_sq->defined()) {
        state->max_exp_avg_sq(*max_exp_avg_sq);
    }

    /*
     * The torch::Tensor assignments above shallow-copy their intrusive
     * handles, so Rust may drop its Tensor arguments after this function.
     */
    adam.state()[parameter_key] = std::move(state);

}
