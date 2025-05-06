import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH     = "/Users/kashmithnisakya/Developer/Data_Science/Energy-Prediction-System/ml/data/processed/china_mill_data_2025_03_04_09_30_30.csv"
TIME_COL      = "time"      # replace with your time column name
VALUE_COL     = "energy"         # replace with your value column name
SAVE_DIR      = "models"
BATCH_SIZE    = 8
EPOCHS        = 50
ENC_LEN       = 3360             # one week of inputs
DEC_LEN       = 3360             # one week of targets
LATENT_UNITS  = 64
TEST_SPLIT    = 0.2

os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── 1. Load & preprocess ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COL])
series = df[VALUE_COL].values.astype("float32")

# normalize (optional but often beneficial)
mean, std = series.mean(), series.std()
series = (series - mean) / std

# build sliding windows of (past_week → next_week)
X, y = [], []
total_len = len(series)
for i in range(total_len - ENC_LEN - DEC_LEN + 1):
    X.append(series[i : i + ENC_LEN])
    y.append(series[i + ENC_LEN : i + ENC_LEN + DEC_LEN])
X = np.array(X)[..., np.newaxis]   # shape (samples, ENC_LEN, 1)
y = np.array(y)[..., np.newaxis]   # shape (samples, DEC_LEN, 1)

# train/test split
n_train = int(X.shape[0] * (1 - TEST_SPLIT))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

logger.info(f"Training on {X_train.shape[0]} samples; testing on {X_test.shape[0]} samples")

# ─── 2. Build seq-to-seq model ────────────────────────────────────────────────
# Encoder
encoder_inputs = Input(shape=(ENC_LEN, 1), name="encoder_inputs")
encoder_lstm1  = LSTM(LATENT_UNITS, return_sequences=True, return_state=True, name="enc_lstm1")
enc_out_seq, enc_h1, enc_c1 = encoder_lstm1(encoder_inputs)
encoder_lstm2  = LSTM(LATENT_UNITS, return_state=True, return_sequences=False, name="enc_lstm2")
_, enc_h2, enc_c2 = encoder_lstm2(enc_out_seq)

# Prepare decoder inputs (we’ll let it learn to map from zeros)
decoder_inputs = tf.zeros_like(encoder_inputs, name="decoder_inputs")  # same shape
# Decoder
decoder_lstm1 = LSTM(LATENT_UNITS, return_sequences=True, return_state=True, name="dec_lstm1")
dec_seq1, _, _ = decoder_lstm1(decoder_inputs, initial_state=[enc_h1, enc_c1])
decoder_lstm2 = LSTM(LATENT_UNITS, return_sequences=True, name="dec_lstm2")
dec_seq2 = decoder_lstm2(dec_seq1, initial_state=[enc_h2, enc_c2])
decoder_dense = TimeDistributed(Dense(1), name="dec_dense")
decoder_outputs = decoder_dense(dec_seq2)

model = Model([encoder_inputs], decoder_outputs, name="seq2seq_model")
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ─── 3. Callbacks ─────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(SAVE_DIR, "energy_seq2seq_epoch_{epoch:02d}.h5"),
    save_freq="epoch",
    save_weights_only=False,
    verbose=1
)

# ─── 4. Train ─────────────────────────────────────────────────────────────────
history = model.fit(
    x= X_train, y= y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb]
)

# ─── 5. Plot training history ────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# ─── 6. Generate & plot predictions ───────────────────────────────────────────
# pick first sample from test set
x_input      = X_test[0:1]
y_true       = y_test[0, :, 0]
y_pred_full  = model.predict(x_input)    # shape (1, DEC_LEN, 1)
y_pred       = y_pred_full[0, :, 0]

plt.figure(figsize=(12, 4))
plt.plot(y_true, label="True Next Week")
plt.plot(y_pred, label="Predicted Next Week")
plt.title("Seq2Seq Prediction: True vs. Predicted")
plt.xlabel("Time Step (hourly)")
plt.ylabel("Normalized Energy")
plt.legend()
plt.grid(True)
plt.show()
