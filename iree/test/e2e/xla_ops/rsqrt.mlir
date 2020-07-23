func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.rsqrt"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 0.707107, 0.57735, 0.5]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<16.0> : tensor<f32>
  %result = "mhlo.rsqrt"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<0.25> : tensor<f32>) : tensor<f32>
  return
}
