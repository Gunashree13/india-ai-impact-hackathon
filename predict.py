"""
ML Inference — drop-in replacement for ml/inference/predict.py
Same interface: predict(filepath) → {"classification": ..., "confidenceScore": ...}
"""
import os, io, json, numpy as np, librosa, torch, warnings
warnings.filterwarnings('ignore')
from joblib import load
from transformers import Wav2Vec2Processor, Wav2Vec2Model

SR = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Prefer project-local ml paths, but fall back to other common locations
MODELS_DIR = None
for p in [
    os.path.join(BASE_DIR, "ml", "models"),
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "..", "models"),
]:
    if os.path.exists(p):
        MODELS_DIR = p
        break
if MODELS_DIR is None:
    MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

FEATURES_DIR = None
for p in [
    os.path.join(BASE_DIR, "ml", "features"),
    os.path.join(BASE_DIR, "features"),
    os.path.join(BASE_DIR, "..", "features"),
]:
    if os.path.exists(p):
        FEATURES_DIR = p
        break
if FEATURES_DIR is None:
    FEATURES_DIR = os.path.join(BASE_DIR, "ml", "features")

_m = {}

def _load():
    if _m: return
    print("Loading models...")
    _m['proc'] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    _m['w2v'] = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    _m['w2v'].eval()
    _m['dev'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _m['w2v'] = _m['w2v'].to(_m['dev'])
    # Prefer the (possibly misspelled) filename requested by users, but fall back
    clf_candidates = [
        os.path.join(MODELS_DIR, "voice_classifer.joblib"),
        os.path.join(MODELS_DIR, "voice_classifier.joblib"),
    ]
    clf_path = None
    for p in clf_candidates:
        if os.path.exists(p):
            clf_path = p
            break
    if clf_path is None:
        raise FileNotFoundError(f"No classifier found in {MODELS_DIR}; looked for: {clf_candidates}")
    _m['clf'] = load(clf_path)

    scaler_path = os.path.join(MODELS_DIR, "signal_scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    _m['scaler'] = load(scaler_path)
    with open(os.path.join(FEATURES_DIR, "signal_feature_names.json")) as f:
        _m['names'] = json.load(f)
    print("✅ Models loaded")

def _embed(audio):
    if len(audio) > SR*10: audio = audio[:SR*10]
    inp = _m['proc'](audio, sampling_rate=SR, return_tensors="pt", padding=True)
    inp = {k: v.to(_m['dev']) for k, v in inp.items()}
    with torch.no_grad():
        out = _m['w2v'](**inp)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def _signal(y):
    if np.max(np.abs(y))>0: y=y/np.max(np.abs(y))
    f={}
    mfccs=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=20)
    for i in range(20):
        v=mfccs[i]; m,s=np.mean(v),np.std(v)+1e-8
        f[f'mfcc_{i}_mean']=np.mean(v); f[f'mfcc_{i}_std']=np.std(v)
        f[f'mfcc_{i}_skew']=float(np.mean(((v-m)/s)**3))
        f[f'mfcc_{i}_kurt']=float(np.mean(((v-m)/s)**4))
    d1=librosa.feature.delta(mfccs); d2=librosa.feature.delta(mfccs,order=2)
    for i in range(20):
        f[f'delta_{i}_mean']=np.mean(d1[i]); f[f'delta_{i}_std']=np.std(d1[i])
        f[f'delta2_{i}_mean']=np.mean(d2[i]); f[f'delta2_{i}_std']=np.std(d2[i])
    try:
        f0,_,vp=librosa.pyin(y,fmin=80,fmax=600,sr=SR); f0v=f0[~np.isnan(f0)]
        if len(f0v)>2:
            f['pitch_mean']=np.mean(f0v);f['pitch_std']=np.std(f0v);f['pitch_min']=np.min(f0v)
            f['pitch_max']=np.max(f0v);f['pitch_range']=np.max(f0v)-np.min(f0v)
            f['pitch_cv']=np.std(f0v)/(np.mean(f0v)+1e-8);pd=np.diff(f0v)
            f['pitch_diff_mean']=np.mean(np.abs(pd));f['pitch_diff_std']=np.std(pd)
            f['pitch_diff_max']=np.max(np.abs(pd));f['jitter']=np.mean(np.abs(pd))/(np.mean(f0v)+1e-8)
            f['voiced_ratio']=np.sum(~np.isnan(f0))/len(f0);f['voiced_prob']=np.mean(vp[~np.isnan(vp)])
        else: raise ValueError()
    except:
        for k in ['pitch_mean','pitch_std','pitch_min','pitch_max','pitch_range','pitch_cv',
                   'pitch_diff_mean','pitch_diff_std','pitch_diff_max','jitter','voiced_ratio','voiced_prob']:
            f[k]=0.0
    sc=librosa.feature.spectral_centroid(y=y,sr=SR)[0]
    f['sc_mean']=np.mean(sc);f['sc_std']=np.std(sc);f['sc_cv']=np.std(sc)/(np.mean(sc)+1e-8)
    sb=librosa.feature.spectral_bandwidth(y=y,sr=SR)[0];f['sb_mean']=np.mean(sb);f['sb_std']=np.std(sb)
    sr_=librosa.feature.spectral_rolloff(y=y,sr=SR)[0];f['sr_mean']=np.mean(sr_);f['sr_std']=np.std(sr_)
    sf_=librosa.feature.spectral_flatness(y=y)[0];f['sf_mean']=np.mean(sf_);f['sf_std']=np.std(sf_);f['sf_max']=np.max(sf_)
    scon=librosa.feature.spectral_contrast(y=y,sr=SR)
    for i in range(scon.shape[0]):f[f'scon_{i}_mean']=np.mean(scon[i]);f[f'scon_{i}_std']=np.std(scon[i])
    rms=librosa.feature.rms(y=y)[0];f['rms_mean']=np.mean(rms);f['rms_std']=np.std(rms);f['rms_max']=np.max(rms)
    f['rms_cv']=np.std(rms)/(np.mean(rms)+1e-8);rd=np.diff(rms)
    f['rms_diff_mean']=np.mean(np.abs(rd));f['rms_diff_std']=np.std(rd)
    f['shimmer']=np.mean(np.abs(rd))/(np.mean(rms)+1e-8)
    zcr=librosa.feature.zero_crossing_rate(y)[0];f['zcr_mean']=np.mean(zcr);f['zcr_std']=np.std(zcr)
    f['zcr_cv']=np.std(zcr)/(np.mean(zcr)+1e-8)
    ch=librosa.feature.chroma_stft(y=y,sr=SR)
    for i in range(12):f[f'chroma_{i}_mean']=np.mean(ch[i]);f[f'chroma_{i}_std']=np.std(ch[i])
    f['chroma_std_mean']=np.mean(np.std(ch,axis=1))
    tn=librosa.feature.tonnetz(y=y,sr=SR)
    for i in range(6):f[f'tonnetz_{i}_mean']=np.mean(tn[i]);f[f'tonnetz_{i}_std']=np.std(tn[i])
    rms_s=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0];sil=rms_s<0.02
    f['silence_ratio']=np.mean(sil);cg=np.diff(sil.astype(int))
    st=np.where(cg==1)[0];en=np.where(cg==-1)[0]
    if len(st)>0 and len(en)>0:
        if en[0]<st[0]:en=en[1:]
        n=min(len(st),len(en))
        if n>0:d=en[:n]-st[:n];f['sil_count']=float(n);f['sil_dur_mean']=np.mean(d);f['sil_dur_std']=np.std(d);f['sil_dur_max']=np.max(d)
        else:f['sil_count']=f['sil_dur_mean']=f['sil_dur_std']=f['sil_dur_max']=0.0
    else:f['sil_count']=f['sil_dur_mean']=f['sil_dur_std']=f['sil_dur_max']=0.0
    S=np.abs(librosa.stft(y));freqs=librosa.fft_frequencies(sr=SR)
    lo=np.mean(S[freqs<1000,:]);mi=np.mean(S[(freqs>=1000)&(freqs<4000),:])
    hi=np.mean(S[freqs>=4000,:]);tot=lo+mi+hi+1e-8
    f['low_ratio']=lo/tot;f['mid_ratio']=mi/tot;f['high_ratio']=hi/tot;f['high_to_low']=hi/(lo+1e-8)
    hs=np.mean(S[freqs>=4000,:],axis=1)
    f['hf_smooth']=np.mean(np.abs(np.diff(hs))) if len(hs)>1 else 0;f['hf_var']=np.var(hs)
    harm=librosa.effects.harmonic(y);perc=librosa.effects.percussive(y)
    f['harm_mean']=np.mean(np.abs(harm));f['perc_mean']=np.mean(np.abs(perc))
    f['hnr']=np.mean(np.abs(harm))/(np.mean(np.abs(perc))+1e-8)
    oe=librosa.onset.onset_strength(y=y,sr=SR);tempo=librosa.feature.tempo(onset_envelope=oe,sr=SR)
    f['tempo']=float(tempo[0]);onsets=librosa.onset.onset_detect(y=y,sr=SR,onset_envelope=oe)
    dur=len(y)/SR;f['onset_rate']=len(onsets)/(dur+1e-8);f['onset_count']=float(len(onsets))
    if len(onsets)>2:oi=np.diff(onsets);f['oi_mean']=np.mean(oi);f['oi_std']=np.std(oi);f['oi_cv']=np.std(oi)/(np.mean(oi)+1e-8)
    else:f['oi_mean']=f['oi_std']=f['oi_cv']=0.0
    f['duration']=dur;f['global_rms']=float(np.sqrt(np.mean(y**2)))
    f['crest']=float(np.max(np.abs(y))/(np.sqrt(np.mean(y**2))+1e-8));f['dyn_range']=float(np.max(y)-np.min(y))
    return f

def predict(audio_path):
    _load()
    audio,_=librosa.load(audio_path,sr=SR,mono=True)
    t,_=librosa.effects.trim(audio,top_db=25)
    if len(t)>SR*0.5: audio=t
    emb=_embed(audio)
    sig=_signal(audio)
    names=_m['names']
    arr=np.array([sig.get(k,0.0) for k in names])
    sig_s=_m['scaler'].transform(arr.reshape(1,-1))
    combined=np.nan_to_num(np.concatenate([emb.reshape(1,-1),sig_s],axis=1))
    proba=_m['clf'].predict_proba(combined)[0]
    cls="AI_GENERATED" if proba[1]>0.5 else "HUMAN"
    conf=round(max(0.51,min(0.99,float(max(proba)))),2)
    return {"classification":cls, "confidenceScore":conf, "features":sig}