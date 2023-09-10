import React, { useEffect, useState } from "react";
import * as onnx from "onnxjs";
import axios from "axios";
import Header from "./Header";
import HeroList from "./HeroList";

const ONNXModel = () => {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [outputData, setOutputData] = useState(null);
  const [inputData1, setInputData1] = useState("");
  const [inputData2, setInputData2] = useState("");
  const [inputData3, setInputData3] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null); // Track errors

  useEffect(() => {
    async function loadModelAndRunInference() {
      try {
        setError(null); // Clear previous errors
        const modelPath = "./onnx_model.onnx";
        const modelUrl = "https://127.0.0.1:8080/src/taskpane/onnx_model.onnx";
        const response = await axios.get(modelUrl, {
          responseType: "arraybuffer",
        });
        const modelData = new Uint8Array(response.data);

        const session = new onnx.InferenceSession();
        await session.loadModel(modelData);

        const inputTensor = new onnx.Tensor(new Float32Array(inputData1 * inputData2 * inputData3), "float32", [
          inputData1 * inputData2 * inputData3,
        ]);

        const outputMap = await session.run([inputTensor]);
        const outputTensor = outputMap.values().next().value;
        console.log(`Output tensor: ${outputTensor.data}`);

        const outputData = outputTensor.data;

        setModelLoaded(true);
        setOutputData(outputData);
      } catch (error) {
        console.error("Error loading or running the ONNX model:", error);
        setError("An error occurred while running the model."); // Set error message
      } finally {
        setIsLoading(false);
      }
    }

    if (isLoading) {
      loadModelAndRunInference();
    }
  }, [isLoading, inputData1, inputData2, inputData3]);

  const handleRunInference = () => {
    setIsLoading(true);
    setModelLoaded(false); // Clear previous results
    setError(null); // Clear previous errors
  };

  return (
    <div className="ms-welcome-content">
      <h2>ONNX Model Results</h2>
      <div>
        <label>
          Input 1: <span> </span>
          <input type="text" value={inputData1} onChange={(e) => setInputData1(e.target.value)} />
        </label>
      </div>
      <div>
        <label>
          Input 2: <span> </span>
          <input type="text" value={inputData2} onChange={(e) => setInputData2(e.target.value)} />
        </label>
      </div>
      <div>
        <label>
          Input 3: <span> </span>
          <input type="text" value={inputData3} onChange={(e) => setInputData3(e.target.value)} />
        </label>
      </div>
      <div>
        <button onClick={handleRunInference}>Run Inference</button>
      </div>
      {error && <div className="error-message">{error}</div>} {/* Display error message */}
      {modelLoaded && (
        <div>
          <h3>Result:</h3>
          <pre>{JSON.stringify(outputData, null, 2)}</pre>
        </div>
      )}
      {isLoading && <div>Loading ONNX model...</div>}
    </div>
  );
};

export default ONNXModel;
