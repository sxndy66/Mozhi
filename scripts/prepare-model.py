from pathlib import Path
import json, urllib.request, hashlib
import torch

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'model-source'
RUNTIME_DIR=ROOT/'public'/'model'
MODEL_DIR.mkdir(parents=True,exist_ok=True)
RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
URL='https://api.wandb.ai/files/abdur-ai4bharat/include-no-cnn/2kuznb3t/augs_transformer.pth'
SRC=MODEL_DIR/'augs_transformer.pth'
BIN=RUNTIME_DIR/'include-transformer-full.bin'
META=RUNTIME_DIR/'include-transformer-full.json'

if not SRC.exists():
    print('Downloading official INCLUDE small Transformer checkpoint...')
    urllib.request.urlretrieve(URL,SRC)

ckpt=torch.load(SRC,map_location='cpu',weights_only=False)
state=ckpt['model']
required={
'l1.weight','l1.bias','embedding.position_embeddings.weight','embedding.LayerNorm.weight','embedding.LayerNorm.bias',
'l2.weight','l2.bias'}
for i in range(2):
    required|={f'layers.{i}.attention.self.{x}.{y}' for x in ('query','key','value') for y in ('weight','bias')}
    required|={f'layers.{i}.attention.output.dense.weight',f'layers.{i}.attention.output.dense.bias',
               f'layers.{i}.attention.output.LayerNorm.weight',f'layers.{i}.attention.output.LayerNorm.bias',
               f'layers.{i}.intermediate.dense.weight',f'layers.{i}.intermediate.dense.bias',
               f'layers.{i}.output.dense.weight',f'layers.{i}.output.dense.bias',
               f'layers.{i}.output.LayerNorm.weight',f'layers.{i}.output.LayerNorm.bias'}
missing=[k for k in required if k not in state]
if missing: raise RuntimeError('Missing model tensors: '+','.join(missing))

# Exact runtime order mirrors the supplied checkpoint metadata used by Mozhi.
order=[
'l1.weight','l1.bias','embedding.position_embeddings.weight','embedding.LayerNorm.weight','embedding.LayerNorm.bias']
for i in range(2):
    order += [f'layers.{i}.attention.self.query.weight',f'layers.{i}.attention.self.query.bias',f'layers.{i}.attention.self.key.weight',f'layers.{i}.attention.self.key.bias',f'layers.{i}.attention.self.value.weight',f'layers.{i}.attention.self.value.bias',f'layers.{i}.attention.output.dense.weight',f'layers.{i}.attention.output.dense.bias',f'layers.{i}.attention.output.LayerNorm.weight',f'layers.{i}.attention.output.LayerNorm.bias',f'layers.{i}.intermediate.dense.weight',f'layers.{i}.intermediate.dense.bias',f'layers.{i}.output.dense.weight',f'layers.{i}.output.dense.bias',f'layers.{i}.output.LayerNorm.weight',f'layers.{i}.output.LayerNorm.bias']
order += ['l2.weight','l2.bias']

parts=[]; meta=[]; offset=0
for name in order:
    t=state[name].detach().cpu().contiguous().numpy().astype('float32',copy=False)
    raw=t.tobytes(order='C')
    parts.append(raw)
    meta.append({'name':name,'shape':list(t.shape),'offset':offset,'length':t.size})
    offset += len(raw)
BIN.write_bytes(b''.join(parts))
META.write_text(json.dumps({'version':2,'dtype':'float32','bytes':offset,'weights':meta,'architecture':{'inputSize':134,'hiddenSize':256,'heads':4,'layers':2,'sequenceLength':169,'classes':263}},indent=2))
print('Prepared',BIN,'bytes=',BIN.stat().st_size)
print('sha256=',hashlib.sha256(BIN.read_bytes()).hexdigest())
