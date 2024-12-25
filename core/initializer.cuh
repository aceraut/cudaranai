#pragma once

#include "common.cuh"

namespace nnv2 {

class Initializer {
public:
  Initializer() {}
  virtual void initialize(Array *a, int fan_in, int fan_out) const = 0;
};

class LecunNormal : public Initializer {
public:
  LecunNormal() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class XavierNormal : public Initializer {
public:
  XavierNormal() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class KaimingNormal : public Initializer {
public:
  KaimingNormal() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class SimplyNormal : public Initializer {
public:
  SimplyNormal() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class LecunUniform : public Initializer {
public:
  LecunUniform() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class XavierUniform : public Initializer {
public:
  XavierUniform() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class KaimingUniform : public Initializer {
public:
  KaimingUniform() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

class SimplyUniform : public Initializer {
public:
  SimplyUniform() : Initializer() {}
  void initialize(Array *a, int fan_in, int fan_out) const override;
};

} // namespace nnv2