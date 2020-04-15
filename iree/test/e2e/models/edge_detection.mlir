// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="1x28x28x1xf32" | IreeFileCheck %s

// Image edge detection module generated by iree/colab/edge_detection.ipynb.
//
// Input : a single 128x128 pixel image as a tensor<1x128x128x1xf32>, with pixels in [0.0, 1.0]
// Output: a single image in the same format after running edge detection

module {
  // CHECK-LABEL: EXEC @edge_detect_sobel_operator
  func @edge_detect_sobel_operator(%arg0: tensor<1x128x128x1xf32> {tf_saved_model.index_path = [0]}) -> (tensor<1x128x128x1xf32> {tf_saved_model.index_path = []}) attributes {iree.module.export, tf._input_shapes = ["tfshape$dim { size: 1 } dim { size: 128 } dim { size: 128 } dim { size: 1 }"]} {
    %0 = xla_hlo.constant dense<[[[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]], [[[-2.000000e+00]], [[0.000000e+00]], [[2.000000e+00]]], [[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>
    %1 = xla_hlo.constant dense<[[[[1.000000e+00]], [[2.000000e+00]], [[1.000000e+00]]], [[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]], [[[-1.000000e+00]], [[-2.000000e+00]], [[-1.000000e+00]]]]> : tensor<3x3x1x1xf32>
    %2 = "xla_hlo.convolution"(%arg0, %0) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x128x128x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x128x128x1xf32>
    %3 = xla_hlo.multiply %2, %2 : tensor<1x128x128x1xf32>
    %4 = "xla_hlo.convolution"(%arg0, %1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x128x128x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x128x128x1xf32>
    %5 = xla_hlo.multiply %4, %4 : tensor<1x128x128x1xf32>
    %6 = xla_hlo.add %3, %5 : tensor<1x128x128x1xf32>
    %7 = "xla_hlo.sqrt"(%6) : (tensor<1x128x128x1xf32>) -> tensor<1x128x128x1xf32>
    return %7 : tensor<1x128x128x1xf32>
  }
  // CHECK: 1x128x128x1xf32=
}