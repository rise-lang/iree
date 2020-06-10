// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse %s | IreeFileCheck %s


module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                     %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_padding"} {
    linalg.conv(%arg0, %arg1, %arg2)
      {dilations = [1, 1],
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
// CHECK-LABEL: func @conv_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   local_size = dense<[32, 1, 1]>
//       CHECK:   scf.parallel (%{{.+}})
//       CHECK:     %[[VIEW1:.+]] = subview %[[ARG1]]
//       CHECK:     %[[VIEW2:.+]] = subview %[[ARG2]]
//       CHECK:     linalg.conv
//  CHECK-SAME:       %[[VIEW1]]
//  CHECK-SAME:       %[[VIEW2]]
//  CHECK-SAME:       "workitem"

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                        %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_no_padding"} {
    linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
// CHECK-LABEL: func @conv_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   local_size = dense<[32, 2, 2]>
//       CHECK:   scf.parallel (%{{.+}}, %{{.+}}, %{{.+}})
//       CHECK:     %[[VIEW1:.+]] = subview %[[ARG1]]
//       CHECK:     %[[VIEW2:.+]] = subview %[[ARG2]]
//       CHECK:     linalg.conv
//  CHECK-SAME:       %[[VIEW1]]
//  CHECK-SAME:       %[[VIEW2]]
//  CHECK-SAME:       "workitem"

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul(%arg0: memref<?x?xf32>,
                %arg1: memref<?x?xf32>,
                %ret0: memref<?x?xf32>) {
    linalg.matmul(%arg0, %arg1, %ret0) :
      memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
    return
  }
}

// CHECK-LABEL: func @matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   local_size = dense<[8, 8, 1]>
//       CHECK:   scf.parallel (%{{.+}}, %{{.+}}, %{{.+}})
//       CHECK:     %[[VIEW1:.+]] = subview %[[ARG1]]
//       CHECK:     %[[VIEW2:.+]] = subview %[[ARG2]]
//       CHECK:     "workitem"
