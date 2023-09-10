console.log("Script running");
const sess = new onnx.InferenceSession();
await sess.loadModel("./onnx_model.onnx");
const input = new onnx.Tensor(new Float32Array(280 * 280 * 4), "float32", [280 * 280 * 4]);
const outputMap = await sess.run([input]);
const outputTensor = outputMap.values().next().value;
console.log(`Output tensor: ${outputTensor.data}`);
