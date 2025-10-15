const tf = require('@tensorflow/tfjs-node');

async function loadModel() {
  // Path to the model.json file.
  const modelPath = 'file://D:/projects/FER-FYP-1-/code/FER-Grp-16-project/tfjs_model/model.json';
  
  // Load the model.
  const model = await tf.loadLayersModel(modelPath);
  
  console.log('Model loaded successfully.');
  
  // You can now use the model to make predictions on new data.
  // For example (replace with your actual data preparation):
  // const input = tf.randomNormal([1, 48, 48, 1]);
  // const prediction = model.predict(input);
  // prediction.print();
}

loadModel();
