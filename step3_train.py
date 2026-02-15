"""
STEP 3: Train combined model. Wav2Vec2 (768) + Signal (268) = 1036 → Ensemble
Usage: python3 step3_train.py
"""
import os, json, numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEAT = "ml/features"
MODELS = "ml/models"
os.makedirs(MODELS, exist_ok=True)

X_embed = np.load(f"{FEAT}/X_embeddings.npy")
X_signal = np.load(f"{FEAT}/X_signal.npy")
y = np.load(f"{FEAT}/y_labels.npy")
print(f"Embed: {X_embed.shape}, Signal: {X_signal.shape}, Human:{sum(y==0)}, AI:{sum(y==1)}")

# Scale signal, combine
scaler = StandardScaler()
X_sig_s = scaler.fit_transform(X_signal)
X = np.nan_to_num(np.concatenate([X_embed, X_sig_s], axis=1))
print(f"Combined: {X.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train:{len(X_tr)} Test:{len(X_te)}")

sp = sum(y_tr==0)/max(sum(y_tr==1),1)
est = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)),
]
try:
    from xgboost import XGBClassifier
    est.append(('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=sp, random_state=42, eval_metric='logloss', use_label_encoder=False)))
    print("XGBoost ✅")
except: print("XGBoost not installed")
try:
    from lightgbm import LGBMClassifier
    est.append(('lgbm', LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1)))
    print("LightGBM ✅")
except: print("LightGBM not installed")

print("\nCV:")
for n,e in est:
    s=cross_val_score(e, X_tr, y_tr, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  {n:8s}: {s.mean():.4f} (+/- {s.std():.4f})")

ens = VotingClassifier(estimators=est, voting='soft', n_jobs=-1)
cal = CalibratedClassifierCV(ens, method='isotonic', cv=3)
cal.fit(X_tr, y_tr)

yp = cal.predict(X_te)
acc = accuracy_score(y_te, yp)
cm = confusion_matrix(y_te, yp)
print(f"\n{'='*40}\n  TEST ACCURACY: {acc:.4f}\n{'='*40}")
print(f"              HUMAN  AI\n  HUMAN:      {cm[0][0]:5d}  {cm[0][1]:5d}\n  AI:         {cm[1][0]:5d}  {cm[1][1]:5d}")
print(classification_report(y_te, yp, target_names=['HUMAN','AI_GENERATED']))

print("Retraining on full data...")
fs = StandardScaler()
Xf = np.nan_to_num(np.concatenate([X_embed, fs.fit_transform(X_signal)], axis=1))
ef = VotingClassifier(estimators=est, voting='soft', n_jobs=-1)
cf = CalibratedClassifierCV(ef, method='isotonic', cv=3)
cf.fit(Xf, y)

dump(cf, f"{MODELS}/voice_classifier.joblib")
dump(fs, f"{MODELS}/signal_scaler.joblib")
print(f"\n✅ Saved to {MODELS}/ — Test accuracy: {acc:.1%}")
