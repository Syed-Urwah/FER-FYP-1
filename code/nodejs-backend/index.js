const express = require('express');
const tf = require('@tensorflow/tfjs');
const fetch = require('node-fetch');
const { createCanvas, loadImage } = require('canvas');
const path = require('path');


const app = express();
const port = 4000;

let model;
const modelPath = path.join(__dirname, 'tfjs_model', 'model.json');
const emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

async function loadModel() {
  try {
  const model = await tf.loadLayersModel('http://127.0.0.1:8080/model.json');
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

app.use(express.json());

app.post('/predict', async (req, res) => {
   console.log(model)
  if (!model) {
    return res.status(500).json({ error: 'Model not loaded yet.' });
  }

  const { imageUrl } = req.body;

  if (!imageUrl) {
    return res.status(400).json({ error: 'imageUrl is required.' });
  }

  try {
    const response = await fetch(imageUrl);
    const buffer = await response.buffer();
    const image = await loadImage(buffer);

    const canvas = createCanvas(48, 48);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 48, 48);

    const tensor = tf.browser.fromPixels(canvas, 1)
      .resizeNearestNeighbor([48, 48])
      .toFloat()
      .expandDims();

    const prediction = model.predict(tensor);
    const probabilities = await prediction.data();
    
    const results = Array.from(probabilities)
      .map((probability, i) => ({
        emotion: emotionLabels[i],
        probability: probability
      }));

    res.json(results);

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Failed to make a prediction.' });
  }
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
  loadModel();
});
