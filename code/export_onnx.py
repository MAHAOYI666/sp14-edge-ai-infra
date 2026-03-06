hailo parser onnx "$ONNX" \
  --hw-arch hailo8 \
  --har-path "$BUILD/yolov5.har" \
  --tensor-shapes images=[1,3,800,800] \
  -y

hailo optimize "$BUILD/yolov5.har" \
  --hw-arch hailo8 \
  --calib-set-path "$CALIB_NPY" \
  --output-har-path "$BUILD/yolov5_calib.har"

mkdir -p "$BUILD/hef_out"

hailo compiler "$BUILD/yolov5_calib.har" \
  --hw-arch hailo8 \
  --output-dir "$BUILD/hef_out"

