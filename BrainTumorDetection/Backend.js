import express from 'express';
import multer from 'multer';
import cors from 'cors';
import FormData from 'form-data';
import fetch from 'node-fetch';
import fs from 'fs';

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

app.post('/api/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // Create form data for Python server
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: form,
      headers: form.getHeaders()
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Python server error: ${error}`);
    }

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Backend error:', err);
    res.status(500).json({ 
      error: err.message || "Failed to analyze image",
      details: err.stack 
    });
  }
});

app.listen(3001, () => console.log('Node backend running on 3001'));