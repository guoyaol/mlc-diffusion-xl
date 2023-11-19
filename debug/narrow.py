import tvm
metadata = tvm.ir.load_json("""{
  \"root\": 1, 
  \"nodes\": [
    {
      \"type_key\": \"\"
    }, 
    {
      \"type_key\": \"Map\", 
      \"keys\": [
        \"relax.expr.Constant\"
      ], 
      \"data\": [2]
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [3, 14, 25, 34]
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"13\", 
        \"data\": \"0\", 
        \"span\": \"0\", 
        \"struct_info_\": \"4\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"shape\": \"5\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"12\", 
        \"span\": \"0\", 
        \"struct_info_\": \"11\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [7, 8, 9, 10]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"77\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"77\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"24\", 
        \"data\": \"1\", 
        \"span\": \"0\", 
        \"struct_info_\": \"15\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"shape\": \"16\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"23\", 
        \"span\": \"0\", 
        \"struct_info_\": \"22\", 
        \"values\": \"17\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [18, 19, 20, 21]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"77\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"77\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\", 
        \"values\": \"17\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"4\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"33\", 
        \"data\": \"2\", 
        \"span\": \"0\", 
        \"struct_info_\": \"26\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"shape\": \"27\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"32\", 
        \"span\": \"0\", 
        \"struct_info_\": \"31\", 
        \"values\": \"28\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [29, 30]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"160\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\", 
        \"values\": \"28\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"42\", 
        \"data\": \"3\", 
        \"span\": \"0\", 
        \"struct_info_\": \"35\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"shape\": \"36\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"41\", 
        \"span\": \"0\", 
        \"struct_info_\": \"40\", 
        \"values\": \"37\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [38, 39]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"128\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\", 
        \"values\": \"37\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAIgAQABAAAAAAAAAAEAAAAAAAAATQAAAAAAAABNAAAAAAAAAKRcAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAIgAQABAAAAAAAAAAEAAAAAAAAATQAAAAAAAABNAAAAAAAAAKRcAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9/////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3////9//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f////3//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQABAAAAAAAAAKAAAAAAAAAAgAIAAAAAAAAAAIA/+a1xPwUpZD+sZVc/GFlLPxH5Pz/vOzU/lhgrP2yGIT9QfRg/mvUPPwvoBz/PTQA/4UDyPrez5D6a6Nc+tNTLPsJtwD4ZqrU+l4CrPpvooT4C2pg+HE2QPqg6iD7Mm4A+ItRyPro+ZT7Ya1g+nFBMPrviQD6HGDY+1ugrPgZLIj7sNhk+0qQQPneNCD756QA+wGfzPRTK5T1o79g9zMzMPflXwT03h7Y9VVGsPa2toj0NlJk9wPyQPXjgiD1XOIE9s/tzPcFVZj1Ec1k9SUlNPYHNQT0q9jY9FbosPZIQIz1p8Rk94VQRPaozCT3ihgE9BJD0PMLh5jxx99k8FMbNPFFDwjxjZbc8EyOtPK1zozz7Tpo8Oa2RPBKHiTyd1YE8qCR1PCBuZzzxe1o8J0NOPGO5Qjzf1Dc8U4wtPArXIzzHrBo8wwUSPKzaCTyKJAI8rbn1O8j65zvCANs7isDOO8IvwzuaRLg7zfWtO6M6pDvOCps7iF6SO3Uuijunc4I7Dk92O82HaDvdhVs7Mz5PO2mmQzuetDg7jV8uO3OeJDsNaRs7grcSO3WCCjvwwgI7weT2OigV6TpQC9w6NbzPOlkdxDrmJLk6iMmuOoUCpTqCx5s6rhCTOqjWijpuEoM633p3OtmiaToTkVw6cDpQOoqURDptlTk6yTMvOtVmJTo1Jhw6GWoTOg8rCzobYgM6QRH4Odkw6jkhF905BbnQOQsMxTk+Bro5UJ6vOWHLpTkhhZw5scOTOaR/izn1sYM5Dah4OS6/ajmPnV057TdROdODRTlTdzo5DAkwOSUwJjlC5Bw5gB0UOW3UCzkIAgQ5PT/5OOpN6zhAJN44\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQABAAAAAAAAAIAAAAAAAAAAAAIAAAAAAAAAAIA/+DluP9avXT+sS04/Efk/PwalMj/gPSY/K7MaP5r1Dz/u9gU/zlP5PmAE6D6a6Nc+I+vIPhv4uj7//K0+m+ihPuqqlj4DNYw+CHmCPiLUcj42+GE+7UdSPnyuQz6HGDY+CXQpPj+wHT6QvRI+d40IPt8k/j3Vf+w9ZBTcPczMzD3HlL49eFmxPVcJpT0NlJk9bOqOPUz+hD0RhXc9wVVmPeNXVj1Adkc9GJ05PRW6LD0bvCA9SJMVPcswCz3ihgE9dxHxPNFU4DyowdA8TkPCPJHGtDyJOag8loucPDmtkTwJkIc8ME18PObIajzxe1o8tVBLPA4zPTxQEDA8CtcjPAZ3GDwu4Q08dwcEPK259TtFquQ7FcrUOwoExjuaRLg7gnmrO8yRnzuvfZQ7dS6KO3yWgDsNUm87erReOzM+TzvC2kA7DXczO1MBJzsNaRs72J4QO22UBjvwePo6KBXpOnXm2DpZ18k669O7Oo3Jrjr3pqI6C1yXOtnZjDpuEoM6ovFzOuMBYzokP1M6ipREOp3uNjpEOyo6omkeOhRqEzoBLgk6rE//OeSV7TkhF905k73NOdZ0vzn7KbI5YculOaBImjl1ko85pJqFOQ2oeDmMZGc54lNXOb9gSDlTdzo5KoUtORl5ITkmQxY5bdQLOSgfAjndLPI4jlzhOA==\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.15.dev0\"}
}""")
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def take2(A: T.Buffer((T.int64(77), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1),), "int64"), T_take: T.Buffer((T.int64(1), T.int64(1280)), "float32")):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(1280)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def argmax(A: T.Buffer((T.int64(1), T.int64(77)), "int32"), A_red: T.Buffer((T.int64(1),), "int64")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1),), "int64")
        A_red_temp_v1 = T.alloc_buffer((T.int64(1),), "int32")
        for ax0, k1 in T.grid(T.int64(1), T.int64(77)):
            with T.block("A_red_temp"):
                v_ax0, v_k1 = T.axis.remap("SR", [ax0, k1])
                T.reads(A[v_ax0, v_k1])
                T.writes(A_red_temp_v0[v_ax0], A_red_temp_v1[v_ax0])
                with T.init():
                    A_red_temp_v0[v_ax0] = T.int64(-1)
                    A_red_temp_v1[v_ax0] = -2147483648
                v_A_red_temp_v0: T.int64 = T.Select(A_red_temp_v1[v_ax0] > A[v_ax0, v_k1] or A_red_temp_v1[v_ax0] == A[v_ax0, v_k1] and A_red_temp_v0[v_ax0] < v_k1, A_red_temp_v0[v_ax0], v_k1)
                v_A_red_temp_v1: T.int32 = T.Select(A_red_temp_v1[v_ax0] > A[v_ax0, v_k1], A_red_temp_v1[v_ax0], A[v_ax0, v_k1])
                A_red_temp_v0[v_ax0] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0] = v_A_red_temp_v1
        for ax0 in range(T.int64(1)):
            with T.block("A_red"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(A_red_temp_v0[v_ax0])
                T.writes(A_red[v_ax0])
                A_red[v_ax0] = A_red_temp_v0[v_ax0]


mod_deploy = Module


target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )


print("start transforming")
mod = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

with open("./after_narrow.py", "w") as f:
    print(mod.script(show_meta=True), file=f)