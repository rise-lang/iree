// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//      CHECK: @concatenate
//      CHECK: linalg.indexed_generic {
// CHECK-SAME:   args_in = 2
// CHECK-SAME:   args_out = 1
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction", "reduction", "reduction"]
// CHECK-NEXT: ^{{.+}}(%[[I:.+]]: index, %[[J:.+]]: index, %[[K:.+]]: index, %[[L:.+]]: index,
// CHECK-SAME:         %[[OPERAND1:.+]]: i32, %[[OPERAND2:.+]]: i32):
//      CHECK:   %[[ACC1:.+]] = constant 0 : index
//      CHECK:   %[[DIM1:.+]] = dim %{{.+}}, 1 : tensor<2x2xi32>
//      CHECK:   %[[LB1:.+]] = cmpi "sge", %[[J]], %[[ACC1]] : index
//      CHECK:   %[[ACC2:.+]] = addi %[[ACC1]], %[[DIM1]] : index
//      CHECK:   %[[UB1:.+]] = cmpi "slt", %[[J]], %[[ACC2]] : index
//      CHECK:   %[[COND1:.+]] = and %[[LB1]], %[[UB1]] : i1
//      CHECK:   %[[VAL1:.+]] = select %[[COND1]], %[[OPERAND1]], %{{.+}} : i32
//      CHECK:   %[[DIM2:.+]] = dim %{{.+}}, 1 : tensor<2x3xi32>
//      CHECK:   %[[LB2:.+]] = cmpi "sge", %[[J]], %[[ACC2]] : index
//      CHECK:   %[[ACC3:.+]] = addi %[[ACC2]], %[[DIM2]] : index
//      CHECK:   %[[UB2:.+]] = cmpi "slt", %[[J]], %[[ACC3]] : index
//      CHECK:   %[[COND2:.+]] = and %[[LB2]], %[[UB2]] : i1
//      CHECK:   %[[RES:.+]] = select %[[COND2]], %[[OPERAND2]], %[[VAL1]] : i32
//      CHECK:   linalg.yield %[[RES]] : i32
func @concatenate(%arg0: tensor<2x2xi32>, %arg1: tensor<2x3xi32>) attributes {
  iree.dispatch_fn_name = ""
} {
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {
    dimension = 1
  } : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  return
}
