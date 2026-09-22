"""
IoTID20 multiclass classification (Cat) with a Keras CNN + Multi-Head Attention model.

Pure Keras 3 / TensorFlow version of Enhanced_IoTID20_CNN_MHA_v2.ipynb (no tree models):
  - feature-level deduplication with label-conflict resolution
  - engineered flow features, data-driven removal of constant / correlated features
  - quantile->normal scaling, features ordered by correlation clustering
  - per-feature periodic (PLR) embeddings + [CLS] token
  - multi-scale CNN stem + CNN/Transformer blocks
  - AdamW, warm-up + cosine decay, label smoothing
  - multi-seed ensemble + per-class bias tuned on the validation split

Usage:
    python iotid20_cnn_mha_keras.py --csv /content/IoT.csv --output ./results
"""

# ==============================================================
# 1. Imports and general settings
# ==============================================================

import argparse
import os
import re
import gc
import json
import math
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, ops

from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

parser = argparse.ArgumentParser(description="IoTID20 CNN + Multi-Head Attention (Keras)")
parser.add_argument("--csv", default="/content/IoT.csv", help="path to the IoTID20 CSV")
parser.add_argument("--output", default="iotid20_keras_results", help="folder for models and reports")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024],
                    help="one model is trained per seed and their probabilities are averaged")
parser.add_argument("--no-dedup", action="store_true",
                    help="keep feature-level duplicates (comparable with the original notebook)")
args, _ = parser.parse_known_args()

SEED = args.seeds[0]
CSV_PATH = args.csv
TARGET_COLUMN = "Cat"

OUTPUT_FOLDER = args.output

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

# --- Data cleaning ---------------------------------------------------------
DEDUPLICATE_ON_FEATURES = not args.no_dedup     # dedup after identifier removal (removes leakage)
CONFLICT_STRATEGY = "majority"       # "majority" | "drop" | "keep" for identical features with different labels
CORRELATION_THRESHOLD = 0.995        # drop one feature of each highly correlated pair

# --- Neural network --------------------------------------------------------
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
WARMUP_EPOCHS = 3
PEAK_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
CLASS_WEIGHTING = "none"             # "none" | "sqrt" (mild inverse-frequency weighting)
MONITOR_METRIC = "val_accuracy"      # "val_accuracy" | "val_macro_f1"
EARLY_STOPPING_PATIENCE = 12

D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_BLOCKS = 3
DROPOUT_RATE = 0.10
ATTENTION_DROPOUT = 0.05
N_FREQUENCIES = 16                   # periodic embedding frequencies per feature
FREQUENCY_SIGMA = 0.5

NN_SEEDS = args.seeds                # multi-seed Keras ensemble
TUNE_CLASS_BIAS = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Keras version     :", keras.__version__)


# ==============================================================
# 2. GPU configuration
# ==============================================================

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

    keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled:", keras.mixed_precision.global_policy())
else:
    print("GPU not detected. Using float32.")
    keras.mixed_precision.set_global_policy("float32")


# ==============================================================
# 3. Helper functions
# ==============================================================

def normalize_column_name(column_name):
    column_name = str(column_name).strip()
    column_name = re.sub(r"[^a-zA-Z0-9_]+", "_", column_name)
    return column_name.strip("_")


def clean_predictor_columns(dataframe):
    dataframe = dataframe.copy()

    object_columns = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in object_columns:
        cleaned = (
            dataframe[column]
            .astype("string")
            .str.strip()
            .replace({
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "null": pd.NA
            })
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")
        available = cleaned.notna().sum()

        numeric_ratio = (
            numeric.notna().sum() / available
            if available > 0 else 0
        )

        if numeric_ratio >= 0.95:
            dataframe[column] = numeric
        else:
            dataframe[column] = cleaned.astype(object)

    return dataframe


def find_column(dataframe, *candidates):
    lookup = {column.casefold(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def safe_ratio(numerator, denominator):
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    return numerator / (denominator.abs() + 1.0)


def evaluate_predictions(name, y_true, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced accuracy": balanced_accuracy_score(y_true, predictions),
        "Macro F1": f1_score(y_true, predictions, average="macro"),
        "Weighted F1": f1_score(y_true, predictions, average="weighted"),
        "Log loss": log_loss(
            y_true,
            np.clip(probabilities, 1e-7, 1.0),
            labels=np.arange(probabilities.shape[1])
        )
    }


# ==============================================================
# 4. Load IoTID20
# ==============================================================

df = pd.read_csv(CSV_PATH, low_memory=False)

print("Original shape:", df.shape)

df.columns = [
    normalize_column_name(column)
    for column in df.columns
]

df = df.replace([np.inf, -np.inf], np.nan)

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype("string")
    .str.strip()
)

df = df[
    df[TARGET_COLUMN].notna()
    & (df[TARGET_COLUMN] != "")
].copy()

print("\nClass distribution (raw):")
print(df[TARGET_COLUMN].value_counts())


# ==============================================================
# 5. Separate predictors and target, remove identifiers/leakage
# ==============================================================

columns_to_remove = {
    "sub_cat",
    "label",
    "flow_id",
    "flowid",
    "src_ip",
    "source_ip",
    "dst_ip",
    "destination_ip",
    "timestamp"
}

removed_columns = [
    column
    for column in df.columns
    if column.casefold() in columns_to_remove and column != TARGET_COLUMN
]

df = df.drop(columns=removed_columns, errors="ignore")

print("Removed identifier/leakage columns:")
print(removed_columns)

feature_frame = clean_predictor_columns(df.drop(columns=[TARGET_COLUMN]))

# Any remaining non-numeric predictor would break the numeric pipeline -> factorise it.
for column in feature_frame.columns:
    if not pd.api.types.is_numeric_dtype(feature_frame[column]):
        print(f"Factorising non-numeric column: {column}")
        feature_frame[column] = pd.factorize(feature_frame[column])[0].astype(np.float64)
        feature_frame.loc[feature_frame[column] < 0, column] = np.nan

df = pd.concat([feature_frame, df[[TARGET_COLUMN]]], axis=1)
del feature_frame
gc.collect()


# ==============================================================
# 6. Feature-level deduplication and label-conflict resolution
# ==============================================================
# v1 removed duplicates while Flow_ID / Timestamp / IPs were still present, so
# rows that are identical from the model's point of view were NOT removed.
# That leaks copies of the same flow into train *and* test, and identical
# feature vectors carrying different labels put a hard ceiling on accuracy.

feature_columns = [column for column in df.columns if column != TARGET_COLUMN]

if DEDUPLICATE_ON_FEATURES:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Exact duplicates (features + label) removed:", before - len(df))

    duplicated_features = df.duplicated(subset=feature_columns, keep=False)
    conflicting_rows = int(duplicated_features.sum())
    print("Rows whose feature vector appears with more than one label:", conflicting_rows)

    if conflicting_rows > 0 and CONFLICT_STRATEGY != "keep":
        conflicts = df[duplicated_features]
        clean = df[~duplicated_features]

        if CONFLICT_STRATEGY == "majority":
            # Majority label within each identical-feature group (count taken
            # before exact-dup removal would be better, but this is a close proxy).
            resolved = (
                conflicts
                .groupby(feature_columns, dropna=False, sort=False)[TARGET_COLUMN]
                .agg(lambda labels: labels.value_counts().index[0])
                .reset_index()
            )
            df = pd.concat([clean, resolved[df.columns]], ignore_index=True)
        elif CONFLICT_STRATEGY == "drop":
            df = clean.reset_index(drop=True)

        print(f"Conflicts resolved with strategy '{CONFLICT_STRATEGY}'.")
else:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Exact duplicates removed:", before - len(df))

print("Shape after cleaning:", df.shape)
print("\nClass distribution (clean):")
print(df[TARGET_COLUMN].value_counts())


# ==============================================================
# 7. Feature engineering
# ==============================================================

def add_engineered_features(frame):
    frame = frame.copy()
    new = {}

    fwd_pkts = find_column(frame, "Tot_Fwd_Pkts", "Total_Fwd_Packets")
    bwd_pkts = find_column(frame, "Tot_Bwd_Pkts", "Total_Backward_Packets")
    fwd_bytes = find_column(frame, "TotLen_Fwd_Pkts", "Total_Length_of_Fwd_Packets")
    bwd_bytes = find_column(frame, "TotLen_Bwd_Pkts", "Total_Length_of_Bwd_Packets")
    duration = find_column(frame, "Flow_Duration")
    src_port = find_column(frame, "Src_Port", "Source_Port")
    dst_port = find_column(frame, "Dst_Port", "Destination_Port")
    init_fwd_win = find_column(frame, "Init_Fwd_Win_Byts", "Init_Win_bytes_forward")
    init_bwd_win = find_column(frame, "Init_Bwd_Win_Byts", "Init_Win_bytes_backward")

    if fwd_pkts and bwd_pkts:
        total_pkts = frame[fwd_pkts] + frame[bwd_pkts]
        new["FE_Total_Pkts"] = total_pkts
        new["FE_Fwd_Bwd_Pkt_Ratio"] = safe_ratio(frame[fwd_pkts], frame[bwd_pkts])
        new["FE_Is_Unidirectional"] = (frame[bwd_pkts] == 0).astype(np.float32)

    if fwd_bytes and bwd_bytes:
        total_bytes = frame[fwd_bytes] + frame[bwd_bytes]
        new["FE_Total_Bytes"] = total_bytes
        new["FE_Fwd_Bwd_Byte_Ratio"] = safe_ratio(frame[fwd_bytes], frame[bwd_bytes])

        if fwd_pkts and bwd_pkts:
            new["FE_Bytes_Per_Pkt"] = safe_ratio(total_bytes, total_pkts)
            new["FE_Fwd_Bytes_Per_Pkt"] = safe_ratio(frame[fwd_bytes], frame[fwd_pkts])
            new["FE_Bwd_Bytes_Per_Pkt"] = safe_ratio(frame[bwd_bytes], frame[bwd_pkts])

    if duration and fwd_pkts and bwd_pkts:
        new["FE_Duration_Per_Pkt"] = safe_ratio(frame[duration], total_pkts)

    for name, column in (("Src", src_port), ("Dst", dst_port)):
        if column:
            port = frame[column]
            new[f"FE_{name}_Port_WellKnown"] = (port < 1024).astype(np.float32)
            new[f"FE_{name}_Port_Registered"] = ((port >= 1024) & (port < 49152)).astype(np.float32)
            new[f"FE_{name}_Port_Dynamic"] = (port >= 49152).astype(np.float32)

    if src_port and dst_port:
        new["FE_Min_Port"] = np.minimum(frame[src_port], frame[dst_port])

    if init_fwd_win and init_bwd_win:
        new["FE_Init_Win_Diff"] = frame[init_fwd_win] - frame[init_bwd_win]

    flag_columns = [column for column in frame.columns if column.endswith("_Flag_Cnt")]
    if flag_columns:
        new["FE_Total_Flags"] = frame[flag_columns].sum(axis=1)

    engineered = pd.DataFrame(new, index=frame.index)
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return pd.concat([frame, engineered], axis=1)


y_text = df[TARGET_COLUMN].copy()
X = add_engineered_features(df.drop(columns=[TARGET_COLUMN]))

print("Features after engineering:", X.shape[1])
print([column for column in X.columns if column.startswith("FE_")])


# ==============================================================
# 8. Stratified 80/10/10 split
# ==============================================================

X_train, X_temp, y_train_text, y_temp_text = train_test_split(
    X,
    y_text,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    stratify=y_text,
    random_state=SEED
)

X_validation, X_test, y_validation_text, y_test_text = train_test_split(
    X_temp,
    y_temp_text,
    test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
    stratify=y_temp_text,
    random_state=SEED
)

del X_temp, y_temp_text
gc.collect()

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)


# ==============================================================
# 9. Remove constant and highly correlated features (train only)
# ==============================================================

constant_columns = [
    column
    for column in X_train.columns
    if X_train[column].nunique(dropna=True) <= 1
]

X_train = X_train.drop(columns=constant_columns)
X_validation = X_validation.drop(columns=constant_columns)
X_test = X_test.drop(columns=constant_columns)

print("Constant columns removed:", constant_columns)

sample = X_train.sample(n=min(200_000, len(X_train)), random_state=SEED)
sample = sample.fillna(sample.median())
correlation = sample.corr(method="pearson").abs().fillna(0.0)
upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))

correlated_columns = [
    column
    for column in upper.columns
    if (upper[column] > CORRELATION_THRESHOLD).any()
]

X_train = X_train.drop(columns=correlated_columns)
X_validation = X_validation.drop(columns=correlated_columns)
X_test = X_test.drop(columns=correlated_columns)

print(f"\nHighly correlated columns removed (|r| > {CORRELATION_THRESHOLD}):")
print(correlated_columns)
print("\nRemaining features:", X_train.shape[1])

del sample, correlation, upper
gc.collect()


# ==============================================================
# 10. Encode target labels
# ==============================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_validation = label_encoder.transform(y_validation_text)
y_test = label_encoder.transform(y_test_text)

NUMBER_OF_CLASSES = len(label_encoder.classes_)
CLASS_NAMES = list(label_encoder.classes_)

print("Classes:")
for i, class_name in enumerate(CLASS_NAMES):
    print(i, class_name, int((y_train == i).sum()))


# ==============================================================
# 11. Neural-network preprocessing
#     median impute + missing indicators + quantile -> normal
# ==============================================================

missing_columns = [
    column
    for column in X_train.columns
    if X_train[column].isna().any()
]

print("Columns with missing values (indicator added):", missing_columns)


class NeuralPreprocessor:

    def __init__(self, seed):
        self.imputer = SimpleImputer(strategy="median")
        self.quantile = QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            subsample=500_000,
            random_state=seed
        )
        self.columns = None
        self.missing_columns = None
        self.order = None

    def fit(self, frame, missing_columns):
        self.columns = list(frame.columns)
        self.missing_columns = list(missing_columns)
        values = self.imputer.fit_transform(frame[self.columns])
        self.quantile.fit(values)
        return self

    def transform(self, frame):
        values = self.imputer.transform(frame[self.columns])
        values = self.quantile.transform(values)
        if self.missing_columns:
            indicators = frame[self.missing_columns].isna().to_numpy(np.float32) * 2.0 - 1.0
            values = np.hstack([values, indicators])
        values = values.astype(np.float32)
        if self.order is not None:
            values = values[:, self.order]
        return values

    def feature_names(self):
        names = self.columns + [f"{column}_missing" for column in self.missing_columns]
        if self.order is not None:
            names = [names[i] for i in self.order]
        return names


neural_preprocessor = NeuralPreprocessor(SEED).fit(X_train, missing_columns)

X_train_nn = neural_preprocessor.transform(X_train)

# --- Order features so that correlated features are neighbours -------------
# Conv1D kernels only make sense if adjacent tokens are related. Hierarchical
# clustering on 1 - |corr| gives such an ordering (fit on training data only).
cluster_sample = X_train_nn[
    np.random.default_rng(SEED).choice(len(X_train_nn), size=min(100_000, len(X_train_nn)), replace=False)
]
feature_correlation = np.nan_to_num(np.corrcoef(cluster_sample, rowvar=False), nan=0.0)
distance = np.clip(1.0 - np.abs(feature_correlation), 0.0, None)
np.fill_diagonal(distance, 0.0)
distance = (distance + distance.T) / 2.0
condensed = squareform(distance, checks=False)
linkage_matrix = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
neural_preprocessor.order = leaves_list(linkage_matrix)

X_train_nn = X_train_nn[:, neural_preprocessor.order]
X_validation_nn = neural_preprocessor.transform(X_validation)
X_test_nn = neural_preprocessor.transform(X_test)

NN_FEATURE_NAMES = neural_preprocessor.feature_names()
NUMBER_OF_FEATURES = X_train_nn.shape[1]

print("Neural-network input features:", NUMBER_OF_FEATURES)

pd.DataFrame({"Feature": NN_FEATURE_NAMES}).to_csv(
    os.path.join(OUTPUT_FOLDER, "model_features.csv"),
    index=False
)

del cluster_sample, feature_correlation, distance, condensed
gc.collect()


# ==============================================================
# 12. Labels, optional class weights and tf.data pipelines
# ==============================================================

y_train_onehot = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES).astype(np.float32)
y_validation_onehot = keras.utils.to_categorical(y_validation, NUMBER_OF_CLASSES).astype(np.float32)
y_test_onehot = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES).astype(np.float32)

class_counts = np.bincount(y_train, minlength=NUMBER_OF_CLASSES).astype(np.float64)

if CLASS_WEIGHTING == "sqrt":
    weights = 1.0 / np.sqrt(class_counts / class_counts.sum())
    weights = weights / (weights * class_counts).sum() * class_counts.sum()
    class_weight = {i: float(w) for i, w in enumerate(weights)}
else:
    class_weight = None

print("Class weights:", class_weight)

AUTOTUNE = tf.data.AUTOTUNE


def make_train_dataset(seed):
    return (
        tf.data.Dataset
        .from_tensor_slices((X_train_nn, y_train_onehot))
        .shuffle(
            buffer_size=min(200_000, len(X_train_nn)),
            seed=seed,
            reshuffle_each_iteration=True
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


validation_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_validation_nn, y_validation_onehot))
    .batch(BATCH_SIZE * 4)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test_nn, y_test_onehot))
    .batch(BATCH_SIZE * 4)
    .prefetch(AUTOTUNE)
)

STEPS_PER_EPOCH = math.ceil(len(X_train_nn) / BATCH_SIZE)
print("Steps per epoch:", STEPS_PER_EPOCH)


# ==============================================================
# 13. Custom layers
#     - PeriodicFeatureEmbedding: per-feature PLR numerical embedding
#       (Gorishniy et al., "On Embeddings for Numerical Features in
#       Tabular Deep Learning", NeurIPS 2022)
#     - CLSToken: learnable classification token
# ==============================================================

@keras.saving.register_keras_serializable(package="IoTID20")
class PeriodicFeatureEmbedding(layers.Layer):

    def __init__(self, d_model, n_frequencies=16, sigma=0.5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.sigma = sigma

    def build(self, input_shape):
        number_of_features = input_shape[-1]

        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(number_of_features, self.n_frequencies),
            initializer=keras.initializers.RandomNormal(stddev=self.sigma),
            trainable=True
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(number_of_features, 2 * self.n_frequencies, self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(number_of_features, self.d_model),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        angles = 2.0 * math.pi * ops.expand_dims(inputs, -1) * self.frequencies
        periodic = ops.concatenate([ops.sin(angles), ops.cos(angles)], axis=-1)
        embedded = ops.einsum("bfk,fkd->bfd", periodic, self.kernel) + self.bias
        return ops.relu(embedded)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_frequencies": self.n_frequencies,
            "sigma": self.sigma
        })
        return config


@keras.saving.register_keras_serializable(package="IoTID20")
class CLSToken(layers.Layer):

    def build(self, input_shape):
        self.token = self.add_weight(
            name="cls_token",
            shape=(1, 1, input_shape[-1]),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        token = ops.broadcast_to(
            ops.cast(self.token, inputs.dtype),
            (batch_size, 1, ops.shape(inputs)[-1])
        )
        return ops.concatenate([token, inputs], axis=1)


# ==============================================================
# 14. CNN + Transformer block (pre-LayerNorm, residual)
# ==============================================================

def cnn_transformer_block(x, block_number, d_model, num_heads, ff_dim, dropout, attention_dropout):
    prefix = f"Block_{block_number}"

    # --- Multi-head self-attention ---
    normalized = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_Attention_LN")(x)
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=attention_dropout,
        name=f"{prefix}_MHA"
    )(normalized, normalized)
    attention = layers.Dropout(dropout, name=f"{prefix}_Attention_Dropout")(attention)
    x = layers.Add(name=f"{prefix}_Attention_Residual")([x, attention])

    # --- Local convolution over neighbouring (correlated) feature tokens ---
    normalized = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_CNN_LN")(x)
    convolution = layers.SeparableConv1D(
        filters=d_model,
        kernel_size=3,
        padding="same",
        activation="gelu",
        name=f"{prefix}_SeparableConv"
    )(normalized)
    convolution = layers.Dropout(dropout, name=f"{prefix}_CNN_Dropout")(convolution)
    x = layers.Add(name=f"{prefix}_CNN_Residual")([x, convolution])

    # --- Position-wise feed-forward (GEGLU-style gating) ---
    normalized = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_FFN_LN")(x)
    value = layers.Dense(ff_dim, name=f"{prefix}_FFN_Value")(normalized)
    gate = layers.Dense(ff_dim, activation="gelu", name=f"{prefix}_FFN_Gate")(normalized)
    feed_forward = layers.Multiply(name=f"{prefix}_FFN_GLU")([value, gate])
    feed_forward = layers.Dropout(dropout, name=f"{prefix}_FFN_Dropout")(feed_forward)
    feed_forward = layers.Dense(d_model, name=f"{prefix}_FFN_Out")(feed_forward)
    x = layers.Add(name=f"{prefix}_FFN_Residual")([x, feed_forward])

    return x


# ==============================================================
# 15. Build the enhanced CNN-MHA model
# ==============================================================

def build_model(number_of_features, number_of_classes):

    inputs = keras.Input(shape=(number_of_features,), name="IoTID20_Input")

    # ---- Tabular MLP branch on the whole feature vector ----
    raw_branch = layers.Dense(256, activation="gelu", name="Raw_Dense_256")(inputs)
    raw_branch = layers.LayerNormalization(name="Raw_LayerNorm")(raw_branch)
    raw_branch = layers.Dropout(DROPOUT_RATE, name="Raw_Dropout")(raw_branch)
    raw_branch = layers.Dense(128, activation="gelu", name="Raw_Dense_128")(raw_branch)

    # ---- Per-feature numerical embeddings -> token sequence ----
    tokens = PeriodicFeatureEmbedding(
        d_model=D_MODEL,
        n_frequencies=N_FREQUENCIES,
        sigma=FREQUENCY_SIGMA,
        dtype="float32",
        name="Periodic_Feature_Embedding"
    )(inputs)

    # ---- Multi-scale CNN stem over correlation-ordered features ----
    conv3 = layers.Conv1D(D_MODEL, 3, padding="same", activation="gelu", name="Stem_Conv3")(tokens)
    conv5 = layers.Conv1D(D_MODEL, 5, padding="same", activation="gelu", name="Stem_Conv5")(tokens)
    conv7 = layers.Conv1D(D_MODEL, 7, padding="same", dilation_rate=1, activation="gelu", name="Stem_Conv7")(tokens)
    multiscale = layers.Concatenate(name="Stem_Concat")([conv3, conv5, conv7])
    multiscale = layers.Dense(D_MODEL, name="Stem_Fuse")(multiscale)
    x = layers.Add(name="Stem_Residual")([tokens, multiscale])

    # ---- Prepend [CLS] and run CNN-Transformer blocks ----
    x = CLSToken(name="CLS_Token")(x)

    for block_number in range(1, NUM_BLOCKS + 1):
        x = cnn_transformer_block(
            x,
            block_number=block_number,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT_RATE,
            attention_dropout=ATTENTION_DROPOUT
        )

    x = layers.LayerNormalization(epsilon=1e-6, name="Final_LN")(x)

    cls_output = layers.Lambda(lambda t: t[:, 0, :], name="CLS_Output")(x)
    feature_tokens = layers.Lambda(lambda t: t[:, 1:, :], name="Feature_Tokens")(x)
    average_pool = layers.GlobalAveragePooling1D(name="Global_Average_Pooling")(feature_tokens)
    maximum_pool = layers.GlobalMaxPooling1D(name="Global_Max_Pooling")(feature_tokens)

    x = layers.Concatenate(name="Combined_Representation")(
        [cls_output, average_pool, maximum_pool, raw_branch]
    )

    x = layers.Dense(256, activation="gelu", name="Classifier_Dense_256")(x)
    x = layers.LayerNormalization(name="Classifier_LN_1")(x)
    x = layers.Dropout(0.20, name="Classifier_Dropout_1")(x)
    x = layers.Dense(128, activation="gelu", name="Classifier_Dense_128")(x)
    x = layers.Dropout(0.10, name="Classifier_Dropout_2")(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        dtype="float32",
        name="Classification_Output"
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Enhanced_IoTID20_CNN_MHA_v2")


build_model(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES).summary()


# ==============================================================
# 16. Compile + callbacks
# ==============================================================

def compile_model(model):
    total_steps = STEPS_PER_EPOCH * EPOCHS
    warmup_steps = STEPS_PER_EPOCH * WARMUP_EPOCHS

    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=PEAK_LEARNING_RATE / 100.0,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.01,
        warmup_target=PEAK_LEARNING_RATE,
        warmup_steps=warmup_steps
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0
    )
    # Do not decay embeddings, biases and normalisation parameters.
    optimizer.exclude_from_weight_decay(
        var_names=["bias", "gamma", "beta", "frequencies", "cls_token"]
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.F1Score(average="macro", name="macro_f1")
        ]
    )
    return model


def make_callbacks(seed):
    checkpoint_path = os.path.join(OUTPUT_FOLDER, f"best_cnn_mha_seed{seed}.weights.h5")
    return checkpoint_path, [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=MONITOR_METRIC,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_FOLDER, f"training_history_seed{seed}.csv"))
    ]


# ==============================================================
# 17. Train CNN-MHA (one model per seed)
# ==============================================================

nn_models = []
histories = []
nn_validation_probabilities = []
nn_test_probabilities = []

for seed in NN_SEEDS:
    print("\n" + "=" * 70)
    print(f"Training CNN-MHA with seed {seed}")
    print("=" * 70)

    set_seed(seed)
    model = compile_model(build_model(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES))
    checkpoint_path, callbacks = make_callbacks(seed)

    history = model.fit(
        make_train_dataset(seed),
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    nn_models.append(model)
    histories.append(history)
    nn_validation_probabilities.append(model.predict(validation_dataset, verbose=0))
    nn_test_probabilities.append(model.predict(test_dataset, verbose=0))

    print(
        f"Seed {seed} validation accuracy:",
        accuracy_score(y_validation, nn_validation_probabilities[-1].argmax(1))
    )

nn_validation_probability = np.mean(nn_validation_probabilities, axis=0)
nn_test_probability = np.mean(nn_test_probabilities, axis=0)

history = histories[0]


# ==============================================================
# 18. Per-class bias calibration on the validation split
# ==============================================================

def apply_bias(probabilities, bias):
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True)


def tune_class_bias(probabilities, y_true, rounds=3):
    # Coordinate ascent on additive per-class log-probability offsets.
    log_probabilities = np.log(np.clip(probabilities, 1e-9, 1.0))
    bias = np.zeros(probabilities.shape[1])
    best_accuracy = accuracy_score(y_true, log_probabilities.argmax(1))
    grid = np.linspace(-2.0, 2.0, 41)

    for _ in range(rounds):
        improved = False
        for class_index in range(len(bias)):
            for value in grid:
                candidate = bias.copy()
                candidate[class_index] = value
                accuracy = accuracy_score(y_true, (log_probabilities + candidate).argmax(1))
                if accuracy > best_accuracy + 1e-9:
                    best_accuracy, bias, improved = accuracy, candidate, True
        if not improved:
            break
    return bias, best_accuracy


CLASS_BIAS = np.zeros(NUMBER_OF_CLASSES)

if TUNE_CLASS_BIAS:
    base_accuracy = accuracy_score(y_validation, nn_validation_probability.argmax(1))
    CLASS_BIAS, tuned_accuracy = tune_class_bias(nn_validation_probability, y_validation)
    print(f"Validation accuracy before bias tuning: {base_accuracy:.6f}")
    print(f"Validation accuracy after  bias tuning: {tuned_accuracy:.6f}")
    print("Class bias:", dict(zip(CLASS_NAMES, np.round(CLASS_BIAS, 3))))

final_test_probability = apply_bias(nn_test_probability, CLASS_BIAS)


# ==============================================================
# 19. Test evaluation (test split used only here)
# ==============================================================

comparison = [
    evaluate_predictions(f"CNN-MHA seed {seed}", y_test, probabilities)
    for seed, probabilities in zip(NN_SEEDS, nn_test_probabilities)
]
if len(NN_SEEDS) > 1:
    comparison.append(evaluate_predictions("CNN-MHA seed ensemble", y_test, nn_test_probability))
comparison.append(evaluate_predictions("Final (ensemble + bias)", y_test, final_test_probability))

comparison = pd.DataFrame(comparison).set_index("Model")
comparison.to_csv(os.path.join(OUTPUT_FOLDER, "model_comparison.csv"))

pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(comparison)

predicted_classes = final_test_probability.argmax(1)

accuracy = accuracy_score(y_test, predicted_classes)
balanced_accuracy = balanced_accuracy_score(y_test, predicted_classes)
macro_f1 = f1_score(y_test, predicted_classes, average="macro")
weighted_f1 = f1_score(y_test, predicted_classes, average="weighted")

print("\n" + "=" * 60)
print(f"Accuracy          : {accuracy:.6f}")
print(f"Balanced Accuracy : {balanced_accuracy:.6f}")
print(f"Macro F1          : {macro_f1:.6f}")
print(f"Weighted F1       : {weighted_f1:.6f}")
print("=" * 60)

report = classification_report(
    y_test,
    predicted_classes,
    target_names=CLASS_NAMES,
    digits=5,
    zero_division=0
)
print(report)

with open(os.path.join(OUTPUT_FOLDER, "classification_report.txt"), "w", encoding="utf-8") as file:
    file.write(report)


# ==============================================================
# 20. Confusion matrices and training curves (saved as PNG)
# ==============================================================

for normalize, fmt, title, filename in (
    (None, "d", "IoTID20 CNN-MHA Confusion Matrix", "confusion_matrix.png"),
    ("true", ".4f", "Normalized IoTID20 CNN-MHA Confusion Matrix", "confusion_matrix_normalized.png"),
):
    cm = confusion_matrix(y_test, predicted_classes, normalize=normalize)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, values_format=fmt, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("loss", "Loss")):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(history.history[metric], label=f"Training {label}")
    ax.plot(history.history[f"val_{metric}"], label=f"Validation {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, f"curve_{metric}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# 21. Save models, preprocessing, encoder, and experiment summary
# ==============================================================

joblib.dump(neural_preprocessor, os.path.join(OUTPUT_FOLDER, "neural_preprocessor.joblib"))
joblib.dump(label_encoder, os.path.join(OUTPUT_FOLDER, "label_encoder.joblib"))

for seed, trained_model in zip(NN_SEEDS, nn_models):
    trained_model.save(os.path.join(OUTPUT_FOLDER, f"iotid20_cnn_mha_seed{seed}.keras"))

with open(os.path.join(OUTPUT_FOLDER, "ensemble_config.json"), "w", encoding="utf-8") as file:
    json.dump(
        {
            "feature_columns": list(X_train.columns),
            "nn_seeds": NN_SEEDS,
            "class_bias": [float(value) for value in CLASS_BIAS],
            "class_names": CLASS_NAMES
        },
        file,
        indent=2
    )

summary = pd.DataFrame({
    "Metric": [
        "Training records",
        "Validation records",
        "Testing records",
        "Input features",
        "Number of classes",
        "Seeds",
        "Accuracy",
        "Balanced accuracy",
        "Macro F1",
        "Weighted F1"
    ],
    "Value": [
        len(X_train),
        len(X_validation),
        len(X_test),
        NUMBER_OF_FEATURES,
        NUMBER_OF_CLASSES,
        " ".join(map(str, NN_SEEDS)),
        accuracy,
        balanced_accuracy,
        macro_f1,
        weighted_f1
    ]
})
summary.to_csv(os.path.join(OUTPUT_FOLDER, "experiment_summary.csv"), index=False)

print(summary)
print("\nIoTID20 Keras CNN-MHA experiment completed.")
print("Results saved to:", OUTPUT_FOLDER)
