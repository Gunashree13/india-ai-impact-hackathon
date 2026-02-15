"""
STEP 2: Extract 268 handcrafted signal features
Usage: python3 step2_extract_signal.py
"""
import os, json, numpy as np, librosa, soundfile as sf, warnings
warnings.filterwarnings('ignore')

SR = 16000
DATA = "data/processed"
OUT = "ml/features"


def extract_features(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if np.max(np.abs(y)) > 0: y = y / np.max(np.abs(y))
    f = {}

    # MFCCs (80)
    mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    for i in range(20):
        v = mfccs[i]; m, s = np.mean(v), np.std(v)+1e-8
        f[f'mfcc_{i}_mean']=np.mean(v); f[f'mfcc_{i}_std']=np.std(v)
        f[f'mfcc_{i}_skew']=float(np.mean(((v-m)/s)**3))
        f[f'mfcc_{i}_kurt']=float(np.mean(((v-m)/s)**4))

    # Deltas (80)
    d1=librosa.feature.delta(mfccs); d2=librosa.feature.delta(mfccs,order=2)
    for i in range(20):
        f[f'delta_{i}_mean']=np.mean(d1[i]); f[f'delta_{i}_std']=np.std(d1[i])
        f[f'delta2_{i}_mean']=np.mean(d2[i]); f[f'delta2_{i}_std']=np.std(d2[i])

    # Pitch (12)
    try:
        f0,_,vp = librosa.pyin(y,fmin=80,fmax=600,sr=SR)
        f0v=f0[~np.isnan(f0)]
        if len(f0v)>2:
            f['pitch_mean']=np.mean(f0v); f['pitch_std']=np.std(f0v)
            f['pitch_min']=np.min(f0v); f['pitch_max']=np.max(f0v)
            f['pitch_range']=np.max(f0v)-np.min(f0v)
            f['pitch_cv']=np.std(f0v)/(np.mean(f0v)+1e-8)
            pd=np.diff(f0v)
            f['pitch_diff_mean']=np.mean(np.abs(pd)); f['pitch_diff_std']=np.std(pd)
            f['pitch_diff_max']=np.max(np.abs(pd))
            f['jitter']=np.mean(np.abs(pd))/(np.mean(f0v)+1e-8)
            f['voiced_ratio']=np.sum(~np.isnan(f0))/len(f0)
            f['voiced_prob']=np.mean(vp[~np.isnan(vp)])
        else: raise ValueError()
    except:
        for k in ['pitch_mean','pitch_std','pitch_min','pitch_max','pitch_range','pitch_cv',
                   'pitch_diff_mean','pitch_diff_std','pitch_diff_max','jitter','voiced_ratio','voiced_prob']:
            f[k]=0.0

    # Spectral (18)
    sc=librosa.feature.spectral_centroid(y=y,sr=SR)[0]
    f['sc_mean']=np.mean(sc); f['sc_std']=np.std(sc); f['sc_cv']=np.std(sc)/(np.mean(sc)+1e-8)
    sb=librosa.feature.spectral_bandwidth(y=y,sr=SR)[0]
    f['sb_mean']=np.mean(sb); f['sb_std']=np.std(sb)
    sr_=librosa.feature.spectral_rolloff(y=y,sr=SR)[0]
    f['sr_mean']=np.mean(sr_); f['sr_std']=np.std(sr_)
    sf_=librosa.feature.spectral_flatness(y=y)[0]
    f['sf_mean']=np.mean(sf_); f['sf_std']=np.std(sf_); f['sf_max']=np.max(sf_)
    scon=librosa.feature.spectral_contrast(y=y,sr=SR)
    for i in range(scon.shape[0]):
        f[f'scon_{i}_mean']=np.mean(scon[i]); f[f'scon_{i}_std']=np.std(scon[i])

    # RMS (7)
    rms=librosa.feature.rms(y=y)[0]
    f['rms_mean']=np.mean(rms); f['rms_std']=np.std(rms); f['rms_max']=np.max(rms)
    f['rms_cv']=np.std(rms)/(np.mean(rms)+1e-8)
    rd=np.diff(rms)
    f['rms_diff_mean']=np.mean(np.abs(rd)); f['rms_diff_std']=np.std(rd)
    f['shimmer']=np.mean(np.abs(rd))/(np.mean(rms)+1e-8)

    # ZCR (3)
    zcr=librosa.feature.zero_crossing_rate(y)[0]
    f['zcr_mean']=np.mean(zcr); f['zcr_std']=np.std(zcr); f['zcr_cv']=np.std(zcr)/(np.mean(zcr)+1e-8)

    # Chroma (25)
    ch=librosa.feature.chroma_stft(y=y,sr=SR)
    for i in range(12): f[f'chroma_{i}_mean']=np.mean(ch[i]); f[f'chroma_{i}_std']=np.std(ch[i])
    f['chroma_std_mean']=np.mean(np.std(ch,axis=1))

    # Tonnetz (12)
    tn=librosa.feature.tonnetz(y=y,sr=SR)
    for i in range(6): f[f'tonnetz_{i}_mean']=np.mean(tn[i]); f[f'tonnetz_{i}_std']=np.std(tn[i])

    # Silence (5)
    rms_s=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0]
    sil=rms_s<0.02; f['silence_ratio']=np.mean(sil)
    chg=np.diff(sil.astype(int)); st=np.where(chg==1)[0]; en=np.where(chg==-1)[0]
    if len(st)>0 and len(en)>0:
        if en[0]<st[0]: en=en[1:]
        n=min(len(st),len(en))
        if n>0:
            d=en[:n]-st[:n]
            f['sil_count']=float(n); f['sil_dur_mean']=np.mean(d)
            f['sil_dur_std']=np.std(d); f['sil_dur_max']=np.max(d)
        else: f['sil_count']=f['sil_dur_mean']=f['sil_dur_std']=f['sil_dur_max']=0.0
    else: f['sil_count']=f['sil_dur_mean']=f['sil_dur_std']=f['sil_dur_max']=0.0

    # High-Freq (9)
    S=np.abs(librosa.stft(y)); freqs=librosa.fft_frequencies(sr=SR)
    lo=np.mean(S[freqs<1000,:]); mi=np.mean(S[(freqs>=1000)&(freqs<4000),:])
    hi=np.mean(S[freqs>=4000,:]); tot=lo+mi+hi+1e-8
    f['low_ratio']=lo/tot; f['mid_ratio']=mi/tot; f['high_ratio']=hi/tot
    f['high_to_low']=hi/(lo+1e-8)
    hs=np.mean(S[freqs>=4000,:],axis=1)
    f['hf_smooth']=np.mean(np.abs(np.diff(hs))) if len(hs)>1 else 0
    f['hf_var']=np.var(hs)
    harm=librosa.effects.harmonic(y); perc=librosa.effects.percussive(y)
    f['harm_mean']=np.mean(np.abs(harm)); f['perc_mean']=np.mean(np.abs(perc))
    f['hnr']=np.mean(np.abs(harm))/(np.mean(np.abs(perc))+1e-8)

    # Tempo (6)
    oe=librosa.onset.onset_strength(y=y,sr=SR)
    tempo=librosa.feature.tempo(onset_envelope=oe,sr=SR)
    f['tempo']=float(tempo[0])
    onsets=librosa.onset.onset_detect(y=y,sr=SR,onset_envelope=oe)
    dur=len(y)/SR; f['onset_rate']=len(onsets)/(dur+1e-8); f['onset_count']=float(len(onsets))
    if len(onsets)>2:
        oi=np.diff(onsets)
        f['oi_mean']=np.mean(oi); f['oi_std']=np.std(oi); f['oi_cv']=np.std(oi)/(np.mean(oi)+1e-8)
    else: f['oi_mean']=f['oi_std']=f['oi_cv']=0.0

    # Global (4)
    f['duration']=dur; f['global_rms']=float(np.sqrt(np.mean(y**2)))
    f['crest']=float(np.max(np.abs(y))/(np.sqrt(np.mean(y**2))+1e-8))
    f['dyn_range']=float(np.max(y)-np.min(y))

    return f


# Get feature names from dummy
dummy = np.zeros(SR, dtype=np.float32)
sf.write("/tmp/_d.wav", dummy, SR) if os.name != 'nt' else sf.write("_d.wav", dummy, SR)
tmp_path = "/tmp/_d.wav" if os.name != 'nt' else "_d.wav"
feature_names = sorted(extract_features(tmp_path).keys())
os.remove(tmp_path)

# Gather files
files, labels = [], []
for name, val in [("human",0),("ai",1)]:
    root = os.path.join(DATA, name)
    if not os.path.exists(root): continue
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(('.wav','.flac','.mp3','.ogg')):
                files.append(os.path.join(dp,fn)); labels.append(val)

print(f"Files: {len(files)}, Features: {len(feature_names)}")
X_list, ys = [], []
for i,(p,l) in enumerate(zip(files,labels)):
    try:
        fd = extract_features(p)
        X_list.append([fd.get(k,0.0) for k in feature_names]); ys.append(l)
        if (i+1)%50==0: print(f"  {i+1}/{len(files)}")
    except Exception as e:
        if i<3: print(f"  Skip: {e}")

X = np.array(X_list); y = np.array(ys)
os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/X_signal.npy", X)
np.save(f"{OUT}/y_signal.npy", y)
with open(f"{OUT}/signal_feature_names.json",'w') as fp: json.dump(feature_names, fp)
print(f"\nDone! {X.shape}, Human:{sum(y==0)}, AI:{sum(y==1)}")
