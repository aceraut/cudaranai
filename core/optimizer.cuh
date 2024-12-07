#pragma once

#include "common.cuh"
#include "layer.cuh"

#include <vector>

namespace nnv2 {

class Optimizer {
public:
  Optimizer() {}

  // Register parameters to the optimizer, you should pass the call
  // to Layer::get_parameters() in the parameter lists
  virtual void add_parameters(std::vector<Param> params) = 0;

  // Update registered parameters
  virtual void update_parameters() = 0;

protected:
  std::vector<Array *> weights;
  std::vector<Array *> grads;
};

class SGD : public Optimizer {
public:
  SGD(float lr = 0.001, float decay = 0.0, float momentum = 0.0)
      : Optimizer(), lr(lr), decay(decay), momentum(momentum) {}

  void add_parameters(std::vector<Param> params) override;
  void update_parameters() override;

private:
  float lr;
  float decay;
  float momentum;

  std::vector<std::unique_ptr<Array>> velocities;
};

class RMSProp : public Optimizer {
public:
  RMSProp(float lr = 0.01, float decay = 0.0, float beta = 0.99)
      : Optimizer(), lr(lr), decay(decay), beta(beta) {}

  void add_parameters(std::vector<Param> params) override;
  void update_parameters() override;

private:
  float lr;
  float decay;
  float beta;

  std::vector<std::unique_ptr<Array>> mean_sqr_grads;
};

class Adam : public Optimizer {
public:
  Adam(
      float lr = 0.001,
      float decay = 0.0,
      float beta1 = 0.9,
      float beta2 = 0.999)
      : Optimizer(), lr(lr), decay(decay), beta1(beta1), beta2(beta2) {}

  void add_parameters(std::vector<Param> params) override;
  void update_parameters() override;

private:
  float lr;
  float decay;
  float beta1;
  float beta2;

  float beta1_pow;
  float beta2_pow;

  std::vector<std::unique_ptr<Array>> mean_grads;
  std::vector<std::unique_ptr<Array>> mean_sqr_grads;
};

} // namespace nnv2