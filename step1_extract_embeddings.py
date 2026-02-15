"""
STEP 1: Extract Wav2Vec2 embeddings (768-dim per sample)
Usage: python3 step1_extract_embeddings.py
"""
import os, numpy as np, librosa, torch, warnings
warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2Model

SR = 16000
DATA = "data/processed"
OUT = "ml/features"

print("Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Device: {device}")

def get_embedding(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) > SR*10: y = y[:SR*10]
    inputs = processor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Gather
files, labels = [], []
for label_name, val in [("human",0), ("ai",1)]:
    root = os.path.join(DATA, label_name)
    if not os.path.exists(root): continue
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(('.wav','.flac','.mp3','.ogg')):
                files.append(os.path.join(dp,fn)); labels.append(val)

print(f"Files: {len(files)}")
feats, ys = [], []
for i,(p,l) in enumerate(zip(files,labels)):
    try:
        feats.append(get_embedding(p)); ys.append(l)
        if (i+1)%50==0: print(f"  {i+1}/{len(files)}")
    except Exception as e:
        if i<3: print(f"  Skip {p}: {e}")

X = np.array(feats); y = np.array(ys)
os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/X_embeddings.npy", X)
np.save(f"{OUT}/y_labels.npy", y)
print(f"\nDone! {X.shape}, Human:{sum(y==0)}, AI:{sum(y==1)}")
