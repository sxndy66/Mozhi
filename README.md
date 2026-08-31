# Mozhi — Deep Learning ISL Translator

Production browser-first Indian Sign Language → text → voice accessibility application.

## Authors
V. Santhosh · J. Vikram Diwakar · J. Prgathesh

3rd Year Artificial Intelligence & Data Science
Sri Balaji Chockalingam Engineering College, Arni

## Architecture
Camera → MediaPipe pose/hand landmarks → INCLUDE preprocessing → temporal sequence → Transformer → 263-class prediction → confidence/stability → text → voice.

## Model
The project uses the supplied AI4Bharat INCLUDE Transformer checkpoint `augs_transformer.pth`. The official INCLUDE implementation documents a small Transformer with 134 input features, hidden size 256, 4 attention heads and 2 Transformer layers, with 263 output classes.

## Deployment
The application is designed for browser-local inference and Vercel deployment. The original PyTorch checkpoint is retained as the source artifact; browser runtime requires a converted inference representation.
