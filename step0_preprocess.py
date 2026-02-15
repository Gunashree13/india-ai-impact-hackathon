"""
STEP 0: Preprocess audio. Works on Windows/Mac/Linux.
Usage: python3 step0_preprocess.py
"""
import os, random, numpy as np, warnings
warnings.filterwarnings('ignore')
import librosa, soundfile as sf
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

SR = 16000

def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y
    except: pass
    if HAS_PYDUB:
        try:
            seg = AudioSegment.from_file(path).set_frame_rate(SR).set_channels(1)
            return np.array(seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        except: pass
    return None

def process(inp, out):
    audio = load_audio(inp)
    if audio is None: return "error"
    if len(audio)/SR < 1.0: return "short"
    try:
        t, _ = librosa.effects.trim(audio, top_db=25)
        if len(t)/SR >= 1.0: audio = t
    except: pass
    if np.sqrt(np.mean(audio**2)) < 0.005: return "silent"
    if len(audio)/SR > 30:
        mx = int(30*SR); s = random.randint(0, len(audio)-mx); audio = audio[s:s+mx]
    pk = np.max(np.abs(audio))
    if pk > 0: audio = audio/pk*0.95
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, audio, SR)
    return "ok"

stats = {"ok":0,"short":0,"silent":0,"error":0}
exts = ('.mp3','.wav','.flac','.ogg','.m4a')
for label in ["ai","human"]:
    inp_dir = os.path.join("data", label)
    out_dir = os.path.join("data/processed", label)
    if not os.path.exists(inp_dir): continue
    files = []
    for dp,_,fns in os.walk(inp_dir):
        if "processed" in dp: continue
        for fn in fns:
            if fn.lower().endswith(exts): files.append(os.path.join(dp,fn))
    print(f"  {label.upper()}: {len(files)} files")
    for i,p in enumerate(files):
        rel = os.path.relpath(p, inp_dir)
        out = os.path.join(out_dir, os.path.splitext(rel)[0]+".wav")
        r = process(p, out); stats[r] += 1
        if (i+1)%200==0: print(f"    {i+1}/{len(files)}")
    print(f"    Done!")
print(f"\nResult: {stats['ok']} ok, {stats['short']} short, {stats['silent']} silent, {stats['error']} errors")
