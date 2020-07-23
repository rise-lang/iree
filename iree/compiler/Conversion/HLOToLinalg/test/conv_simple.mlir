// RUN: iree-opt -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s
func @main(%arg0: tensor<1x5x5x1xf32>, %arg1: tensor<3x3x1x1xf32>) -> tensor<1x5x5x1xf32>
  attributes {iree.module.export} {
    %1 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x5x5x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x5x5x1xf32>
    return %1 : tensor<1x5x5x1xf32>
  }
// correct invocation:
//./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="1x5x5x1xf32=[[[1][2][3][4][5]][[6][7][8][9][10]][[11][12][13][14][15]][[16][17][18][19][20]][[21][22][23][24][25]]], 3x3x1x1xf32=[[[1]][[1]][[1]]][[[1]][[1]][[1]]][[[1]][[1]][[1]]]" -entry_function=main
// but remember to recompile the thing iree-translate!
//
//
// piping does not work, because the module file is too large!
// benchmark a module:
// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir >> /tmp/conv.module
// ./iree-benchmark-module --input_file=/tmp/conv.module --driver=llvm --entry_function=main --inputs="1x5x5x1xf32=[[[1][2][3][4][5]][[6][7][8][9][10]][[11][12][13][14][15]][[16][17][18][19][20]][[21][22][23][24][25]]], 3x3x1x1xf32=[[[1]][[1]][[1]]][[[1]][[1]][[1]]][[[1]][[1]][[1]]]"





// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="1x5x5x1xf32=[[[1][2][3][4][5]][[6][7][8][9][10]][[11][12][13][14][15]][[16][17][18][19][20]][[21][22][23][24][25]]], 3x3x1x1xf32=[[[1]][[1]][[1]]][[[1]][[1]][[1]]][[[1]][[1]][[1]]]" -entry_function=main
// input: 1x5x5x1xf32=[[[1][2][3][4][5]][[6][7][8][9][10]][[11][12][13][14][15]][[16][17][18][19][20]][[21][22][23][24][25]]]
// kernel: 3x3x1x1xf32=[[[1]][[1]][[1]]][[[1]][[1]][[1]]][[[1]][[1]][[1]]]
// result: [[[16][27][33][39][28]][[39][63][72][81][57]][[69][108][117][126][87]][[99][153][162][171][117]][[76][117][123][129][88]]]
//         [[[16][27][33][39][28]][[39][63][72][81][57]][[69][108][117][126][87]][[99][153][162][171][117]][[76][117][123][129][88]]]
//         [[[16][27][33][39][28]][[39][63][72][81][57]][[69][108][117][126][87]][[99][153][162][171][117]][[76][117][123][129][88]]]


//func @main(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
//  attributes {iree.module.export} {
//%2 = "mhlo.convolution"(%arg0, %arg1) {
//      batch_group_count = 1 : i64,
//      dimension_numbers = {
//        input_batch_dimension = 0 : i64,
//        input_feature_dimension = 3 : i64,
//        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
//        kernel_input_feature_dimension = 2 : i64,
//        kernel_output_feature_dimension = 3 : i64,
//        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
//        output_batch_dimension = 0 : i64,
//        output_feature_dimension = 3 : i64,
//        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
//      },
//      feature_group_count = 1 : i64,
//      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
//      rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
//      window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
//    return %2 : tensor<3x5x5x4xf32>
//}
// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="2x2x3x4xf32=[[[1 1 1 1][1 1 1 1][1 1 1 1]][[1 1 1 1][1 1 1 1][1 1 1 1]]][[[1 1 1 1][1 1 1 1][1 1 1 1]][[1 1 1 1][1 1 1 1][1 1 1 1]]], 3x5x5x3xf32=[[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]][[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]][[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]]" -entry_function=main
// kernel: 2x2x3x4xf32=[[[1 1 1 1][1 1 1 1][1 1 1 1]][[1 1 1 1][1 1 1 1][1 1 1 1]]][[[1 1 1 1][1 1 1 1][1 1 1 1]][[1 1 1 1][1 1 1 1][1 1 1 1]]]
// input: 3x5x5x3xf32=[[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]][[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]][[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]]


//func @main(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<1x5x5x3xf32>) -> tensor<1x5x5x4xf32>
//  attributes {iree.module.export} {
//    %2 = "mhlo.convolution"(%arg0, %arg1) {
//      batch_group_count = 1 : i64,
//      dimension_numbers = {
//        input_batch_dimension = 0 : i64,
//        input_feature_dimension = 3 : i64,
//        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
//        kernel_input_feature_dimension = 2 : i64,
//        kernel_output_feature_dimension = 3 : i64,
//        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
//        output_batch_dimension = 0 : i64,
//        output_feature_dimension = 3 : i64,
//        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
//      },
//      feature_group_count = 1 : i64,
//      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
//      rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
//      window_strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<1x2x3x4xf32>, tensor<1x5x5x3xf32>) -> tensor<1x5x5x4xf32>
//    return %2 : tensor<1x5x5x4xf32>
//}
// produces: linalg.conv(%1, %2, %0) {dilations = [1, 2], padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, strides = [2, 1]} : memref<3x5x5x3xf32>, memref<2x2x3x4xf32>, memref<3x5x5x4xf32>
//
//
// iree-opt -iree-transformation-pipeline contains the correct pass order for lowering.
// ./iree-opt -iree-transformation-pipeline -iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir
//
//
// translate to iree module for execution: ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir
// Then use iree-run-mlir to execute it. TODO!

// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="1x2x3x4xf32, 1x5x5x4xf32" -entry_function=main
// input: 1x2x3x4xf32=[[[1 1 1 1][2 2 2 2][3 3 3 3]][[4 4 4 4][5 5 5 5][6 6 6 6]]]
// kernel: 1x5x5x3xf32=[[[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]][[1 1 1][1 1 1][1 1 1][1 1 1][1 1 1]]]

// first kernel then input
//func @simple(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
//  attributes {iree.module.export} {
//    %2 = "mhlo.convolution"(%arg0, %arg1) {
//      batch_group_count = 1 : i64,
//      dimension_numbers = {
//        input_batch_dimension = 0 : i64,
//        input_feature_dimension = 3 : i64,
//        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
//        kernel_input_feature_dimension = 2 : i64,
//        kernel_output_feature_dimension = 3 : i64,
//        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
//        output_batch_dimension = 0 : i64,
//        output_feature_dimension = 3 : i64,
//        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
//      },
//      feature_group_count = 1 : i64,
//      padding = dense<[[0, 2], [0, 2]]> : tensor<2x2xi64>,
//      rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
//      window_strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<1x3x3x1xf32>, tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
//    return %2 : tensor<1x4x4x1xf32>
//}
// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="1x3x3x1xf32=[[[1][1][1]][[1][1][1]][[1][1][1]]], 1x4x4x1xf32=[[[1][2][3][4]][[5][6][7][8]][[9][10][11][12]][[13][14][15][16]]]" -entry_function=simple
// input: 1x4x4x1xf32=[[[1][2][3][4]][[5][6][7][8]][[9][10][11][12]][[13][14][15][16]]]
// kernel: 1x3x3x1xf32=[[[0][0][0]][[0][0][0]][[0][0][0]]]


//
// first kernel then input
//func @simple(%arg0: tensor<3x3x1x1xf32>, %arg1: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
//  attributes {iree.module.export} {
//    %2 = "mhlo.convolution"(%arg0, %arg1) {
//      batch_group_count = 1 : i64,
//      dimension_numbers = {
//        input_batch_dimension = 0 : i64,
//        input_feature_dimension = 1 : i64,
//        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
//        kernel_input_feature_dimension = 1 : i64,
//        kernel_output_feature_dimension = 1 : i64,
//        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
//        output_batch_dimension = 0 : i64,
//        output_feature_dimension = 1 : i64,
//        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
//      },
//      feature_group_count = 1 : i64,
//      padding = dense<[[0, 2], [0, 2]]> : tensor<2x2xi64>,
//      rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
//      window_strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<3x3x1x1xf32>, tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
//    return %2 : tensor<1x4x4x1xf32>
//}
// ./iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=llvm-ir ../../../iree/compiler/Conversion/HLOToLinalg/test/conv_simple.mlir | ./iree-run-module -driver=llvm -inputs="3x3x1x1xf32=[[[[1]]][[[1]]][[[1]]]][[[[1]]][[[1]]][[[1]]]][[[[1]]][[[1]]][[[1]]]], 1x4x4x1xf32=[[[1][2][3][4]][[5][6][7][8]][[9][10][11][12]][[13][14][15][16]]]" -entry_function=simple
// input: 1x4x4x1xf32=[[[1][2][3][4]][[5][6][7][8]][[9][10][11][12]][[13][14][15][16]]]
// kernel: 1x3x3x1xf32=[[[0][0][0]][[0][0][0]][[0][0][0]]]
// kernel: 3x3x1x1xf32=[[[[1]]][[[1]]][[[1]]]][[[[1]]][[[1]]][[[1]]]][[[[1]]][[[1]]][[[1]]]]