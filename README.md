# Mozhi — Deep Learning ISL Translator

Mozhi is a premium browser-first Indian Sign Language accessibility product: live camera → visual landmarks → AI4Bharat INCLUDE Transformer → text → voice.

## Authors
V. Santhosh  
J. Vikram Diwakar  
J. Prgathesh  
3rd Year — Artificial Intelligence & Data Science  
Sri Balaji Chockalingam Engineering College, Arni

## Product
- Real-time front-camera capture with recoverable camera permission states
- MediaPipe hand + pose landmark extraction
- 169-frame temporal feature sequence
- 134 numerical features per frame
- Full AI4Bharat INCLUDE Transformer runtime weights
- 263-class label map
- Confidence + repeated-prediction stability gate
- Text output, sentence controls and browser speech
- Learn, History, Settings and About sections
- Premium forest-green / warm-cream visual system
- Mozhi symbol, wordmark and original hero illustration assets
- Responsive mobile layout and accessibility controls

## Deep-learning pipeline
Camera → MediaPipe hand/pose landmarks → INCLUDE preprocessing → 169 × 134 temporal tensor → 2-layer Transformer → 263-class Softmax → stable prediction → text → voice.

The official INCLUDE source defines a small Transformer configuration with input_size 134, hidden_size 256, 4 attention heads and 2 Transformer layers. The official training runner supports pretrained Transformer checkpoints and the INCLUDE label map defines 263 classes.

## Model files
`model-source/augs_transformer.pth` is the full pretrained INCLUDE checkpoint (source artifact).

`public/model/include-transformer-full.bin` contains the full model tensors required by the browser runtime, not a reduced classifier. `public/model/include-transformer-full.json` describes every tensor, shape and byte offset. `public/model/labels.json` contains the 263 class labels.

A GitHub Actions workflow keeps the source checkpoint and browser runtime artifact reproducible by downloading the official AI4Bharat pretrained checkpoint and regenerating the browser tensor bundle.

## Browser inference
The repository runs the Transformer directly in the browser with TensorFlow.js. The raw PyTorch checkpoint is kept for provenance; the browser uses the generated tensor bundle because JavaScript cannot execute a `.pth` checkpoint directly.

## Deployment
The site is static and requires HTTPS for camera access. No Python inference server or WebSocket inference server is required during runtime. Vercel can serve the site and the model artifacts directly from the repository.
