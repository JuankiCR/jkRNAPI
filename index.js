const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs')

const startServer = async () => {
  const model = await tf.loadLayersModel('https://rn-team4-7b.juankicr.dev/assets/model/model.json')
  const app = express();
  const port = 3062;

  app.use(bodyParser.json());

  app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'https://rn-team4-7b.juankicr.dev');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
  });

  app.get('/', (req, res) => {
    res.send('¡Hola, RN API!');
  });

  app.post('/tomateorapple', (req, res) => {
    try {
      const img_normalized = req.body.img_normalized;

      if (!img_normalized || !Array.isArray(img_normalized) || img_normalized.length !== 32 || !img_normalized.every(row => Array.isArray(row) && row.length === 32)) {
        return res.status(400).json({ error: 'La matriz debe ser de dimensiones 32x32' });
      }

      if (!img_normalized.every(row => row.every(value => typeof value === 'number' && value >= 0 && value <= 1))) {
        return res.status(400).json({ error: 'Los valores de la matriz deben estar en el rango de 0 a 1' });
      }

      const imgTensor = tf.tensor([img_normalized]).expandDims(3);

      const isTomato = model.predict(imgTensor);

      const res_value = isTomato.arraySync()[0];

      res.json({ 
        res_value: res_value[0],
        res_text: res_value[0] < 0.5 ? "Apple!" : "Tomato!"
      });
    } catch (error) {
      console.error('Error:', error);
      res.status(500).json({ error: 'Error interno del servidor' });
    }
  });

  app.listen(port, () => {
    console.log(`La aplicación está escuchando en http://localhost:${port}`);
  });
};

// Llamando a la función asíncrona para iniciar el servidor
startServer();
