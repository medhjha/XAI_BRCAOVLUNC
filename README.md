# XAI_BRCAOVLUNC
XAI-driven comparative transcriptomic analysis of breast, ovarian and lung cancer
# Setup and Imports

print("Installing and Importing Libraries")


!pip install pandas numpy scikit-learn matplotlib seaborn jupyter 
!pip install combat 
!pip install shap 
!pip install lifelines 
!pip install gseapy 
import pandas as pd
import numpy as np
import os
import tarfile
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from combat.pycombat import pycombat
rom sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
print("Libraries installed and imported successfully.")
Define file paths
define base directory for clinical files
CLINICAL_DIR = os.path.dirname(CLINICAL_ARCHIVE_FILE)
CLINICAL_FILE_TSV = "clinical.tsv"
FOLLOWUP_FILE_TSV = "follow_up.tsv"
CLINICAL_FILE_TSV_PATH = os.path.join(CLINICAL_DIR, CLINICAL_FILE_TSV)
FOLLOWUP_FILE_TSV_PATH = os.path.join(CLINICAL_DIR, FOLLOWUP_FILE_TSV)
print(f"Data directory set to: {DATA_DIR}")
print("\n--- Phase 1: Loading Metadata and Gene Counts ---")
f not os.path.exists(CLINICAL_FILE_TSV_PATH) or not os.path.exists(FOLLOWUP_FILE_TSV_PATH):
    print(f"Extracting {CLINICAL_ARCHIVE_FILE}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with tarfile.open(CLINICAL_ARCHIVE_FILE, "r:gz") as tar:
            tar.extractall(path=CLINICAL_DIR)
            print(f"Successfully extracted clinical files to {CLINICAL_DIR}")
else:
    print("Clinical files already extracted.")
    print("Loading metadata...")
manifest_df = pd.read_csv(MANIFEST_FILE, sep="\t")
sample_sheet_df = pd.read_csv(SAMPLE_SHEET_FILE, sep="\t")
manifest_df = manifest_df.rename(columns={'id': 'Manifest_File_ID'})
metadata_df = pd.merge(manifest_df, sample_sheet_df, left_on="filename", right_on="File Name", how="left")
print("Metadata merge successful!")
def get_label(row):
    if pd.notna(row["Tissue Type"]) and "Normal" in row["Tissue Type"]:
        return "Normal"
    elif row["Project ID"] == "TCGA-BRCA":
        return "BRCA"
    elif row["Project ID"] == "TCGA-OV":
        return "OV"
    elif row["Project ID"] in ["TCGA-LUAD", "TCGA-LUSC"]:
        return "LUNG"
    else:
        return "Other"

metadata_df["label"] = metadata_df.apply(get_label, axis=1)
metadata_df["batch"] = metadata_df["Project ID"]
metadata_df = metadata_df[metadata_df["label"] != "Other"].copy()
metadata_df = metadata_df.set_index("Manifest_File_ID")
metadata_df.index.name = "File ID"
print(f"Created metadata for {len(metadata_df)} samples.")
print(metadata_df["label"].value_counts())
count_data_list = []
sample_ids_in_order = []
print("Loading and combining count files... (This may take several minutes)")
counts_df.index = counts_df.index.str.split('.').str[0]
print(f"Genes before deduplication: {len(counts_df)}")
counts_df = counts_df.groupby(counts_df.index).sum()
print(f"Genes after deduplication: {len(counts_df)}")
print(f"\nGenes before filtering: {len(counts_df)}")
min_counts = 10
min_samples_pct = 0.1
min_samples = max(1, int(counts_df.shape[1] * min_samples_pct))
keep_genes = (counts_df >= min_counts).sum(axis=1) >= min_samples
counts_df = counts_df.loc[keep_genes]
print(f"Genes after filtering: {len(counts_df)}")
def calculate_cpm(df):
    total_counts = df.sum(axis=0).replace(0, 1)
    cpm_df = df.apply(lambda x: (x / total_counts[x.name]) * 1_000_000, axis=0)
    return cpm_df

print("\nCalculating CPM...")
cpm_df = calculate_cpm(counts_df)
print("Calculating log2(CPM+1)...")
log_cpm_df = np.log2(cpm_df + 1)
batches = metadata_df['batch'].tolist()
print("\nRunning ComBat for batch correction...")
log_cpm_df = log_cpm_df.fillna(0)
# Correct ComBat call
log_cpm_combat_df = pycombat(log_cpm_df.astype(float), batches)
log_cpm_combat_df = pd.DataFrame(log_cpm_combat_df, index=log_cpm_df.index, columns=log_cpm_df.columns)
print("Batch correction complete.")
print("\nApplying Quantile Normalization...")
qt = QuantileTransformer(output_distribution='normal', random_state=42)
data_transposed = log_cpm_combat_df.T
data_quantiled = qt.fit_transform(data_transposed)
final_data_df = pd.DataFrame(data_quantiled.T, index=log_cpm_combat_df.index, columns=log_cpm_combat_df.columns)
print(f"Final data matrix shape: {final_data_df.shape}")
print("\nGenerating PCA plots...")
X_log_cpm = log_cpm_df.T
X_combat = log_cpm_combat_df.T
plot_labels = metadata_df['label']
plot_batches = metadata_df['batch']
def plot_pca(data, title, color_by, palette='viridis'):
    data_numeric = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler_pca = StandardScaler()
    data_scaled = scaler_pca.fit_transform(data_numeric)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data_numeric.index)
    pc_df['Color'] = color_by.loc[pc_df.index].values
    explained_var = pca.explained_variance_ratio_

  plt.figure(figsize=(11, 8))
    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='Color', palette=palette, s=50, alpha=0.7, edgecolor=None)
    plt.title(f"{title}\nPC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}")
    plt.xlabel(f"PC1 ({explained_var[0]:.1%})")
    plt.ylabel(f"PC2 ({explained_var[1]:.1%})")
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 15:
        plt.legend(handles[:15], labels[:15], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title=color_by.name)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title=color_by.name)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    print("Plotting PCA before batch correction (colored by batch)...")
plot_pca(X_log_cpm, "PCA Before Batch Correction (Colored by Batch)", plot_batches, palette='tab10')
print("Plotting PCA after ComBat (colored by batch)...")
plot_pca(X_combat, "PCA After ComBat Correction (Colored by Batch)", plot_batches, palette='tab10')
print("Plotting PCA after ComBat (colored by class)...")
plot_pca(X_combat, "PCA After ComBat Correction (Colored by Class)", plot_labels, palette='Set2')
X = final_data_df.T
y_strings = metadata_df['label']
label_map = {label: i for i, label in enumerate(y_strings.unique())}
y = y_strings.map(label_map)
class_names_in_order = [label for label, i in sorted(label_map.items(), key=lambda item: item[1])]
print(f"Label mapping created: {label_map}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Splitting data: Training set {X_train.shape}, Test set {X_test.shape}")
ipelines = {
    "KNN": Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=5))]),
    "RF": Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))]),
    "MLP": Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1))])
}
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'mcc': make_scorer(matthews_corrcoef)
}

cv_results = {}
print("\nRunning 5-fold cross-validation...")
for model_name, pipe in pipelines.items():
    print(f"--- Training {model_name} ---")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_splitter, scoring=scoring_metrics, n_jobs=-1, error_score='raise')
    cv_results[model_name] = scores
    print(f"Finished {model_name}.")
    print("\n--- Cross-Validation Results ---")
best_mcc = -1.0
BEST_MODEL_NAME = ""
cv_summary = []

for model_name, scores in cv_results.items():
    mean_acc = np.mean(scores['test_accuracy'])
    mean_f1 = np.mean(scores['test_f1_weighted'])
    mean_mcc = np.mean(scores['test_mcc'])
    cv_summary.append({'Model': model_name, 'Mean Accuracy': mean_acc, 'Mean F1-Weighted': mean_f1, 'Mean MCC': mean_mcc})
    if mean_mcc > best_mcc:
        best_mcc = mean_mcc
        BEST_MODEL_NAME = model_name

cv_summary_df = pd.DataFrame(cv_summary).set_index('Model').round(4)
print("\nCross-Validation Summary:")
print(cv_summary_df)
print(f"\n--- Best Model Selected: {BEST_MODEL_NAME} (MCC: {best_mcc:.4f}) ---")
best_pipeline = pipelines[BEST_MODEL_NAME]
# evaluation for XAI
print(f"Training final {BEST_MODEL_NAME} model on all training data...")
best_pipeline.fit(X_train, y_train)
print("Final model training complete.")
print("\nEvaluating final model on the held-out TEST set...")
y_pred = best_pipeline.predict(X_test)
print("\n--- Test Set Performance ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names_in_order))
final_mcc = matthews_corrcoef(y_test, y_pred)
print(f"Test Set MCC: {final_mcc:.4f}")
cm = confusion_matrix(y_test, y_pred, labels=best_pipeline.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_in_order)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, values_format='d')
plt.title(f"Test Set Confusion Matrix - {BEST_MODEL_NAME}")
plt.tight_layout()
plt.show()
print("\nCalculating SHAP values...")
model = best_pipeline.named_steps['model']
scaler = best_pipeline.named_steps['scaler']
X_train_scaled_shap = scaler.transform(X_train)
X_test_scaled_shap = scaler.transform(X_test)
feature_names = X_train.columns.tolist()
X_test_scaled_shap_df = pd.DataFrame(X_test_scaled_shap, index=X_test.index, columns=feature_names)
X_test_scaled_shap_array = X_test_scaled_shap_df.values
shap_values_raw = None
if BEST_MODEL_NAME == "RF":
    print("Using TreeExplainer for SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test_scaled_shap_array)
else:
    print(f"Using KernelExplainer for SHAP (this is slow)...")
    background_data = shap.sample(X_train_scaled_shap, 100)
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values_raw = explainer.shap_values(X_test_scaled_shap_array[:50], nsamples=100) 
    
  if X_test_scaled_shap_array.shape[0] != shap_values_raw[0].shape[0]:
         X_test_scaled_shap_array = X_test_scaled_shap_array[:50]
         X_test_scaled_shap_df = X_test_scaled_shap_df.iloc[:50]
print("SHAP values calculated.")
shap_values = None
if shap_values_raw is not None and BEST_MODEL_NAME == "RF":
    print("Re-stacking SHAP values for summary plots...")
    try:
        shap_values_stacked = np.array(shap_values_raw)
        n_classes = shap_values_stacked.shape[2]
        shap_values_by_class = []
        for i in range(n_classes):
            shap_values_by_class.append(shap_values_stacked[:, :, i])
        shap_values = shap_values_by_class
        print(f"Re-stacked into a list of {len(shap_values)} arrays.")
    except Exception as e:
        print(f"Error during SHAP re-stacking: {e}")
        shap_values = shap_values_raw
elif shap_values_raw is not None:
    shap_values = shap_values_raw
    if shap_values is not None and isinstance(shap_values, list) and len(shap_values) == len(class_names_in_order):
    print("Generating SHAP Global Bar Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_shap_df, plot_type="bar",
                      class_names=class_names_in_order,
                      feature_names=feature_names,
                      max_display=20,
                      show=False)
    plt.title(f"Global Feature Importance (Top 20) - {BEST_MODEL_NAME}")
    plt.tight_layout()
    plt.show()

   print("\nGenerating SHAP Summary Plots (Beeswarm)...")
    shap_sum_list = []
    for i, class_name in enumerate(class_names_in_order):
        print(f"Plotting SHAP for class: {class_name}")
        plt.figure()
        
  shap_vals = shap_values[i]
        
   if shap_vals.shape[1] > X_test_scaled_shap_array.shape[1]:
            print(f"  Adjusting SHAP shape from {shap_vals.shape[1]} to {X_test_scaled_shap_array.shape[1]} features.")
            shap_vals = shap_vals[:, :X_test_scaled_shap_array.shape[1]]
        elif shap_vals.shape[1] != X_test_scaled_shap_array.shape[1]:
            raise ValueError(f"Feature dimension mismatch for class {class_name}.")
        
   shap_sum_list.append(shap_vals)
        
   shap.summary_plot(
            shap_vals, X_test_scaled_shap_array,
            feature_names=feature_names,
            max_display=20,
            show=False
        )
        plt.title(f"Top 20 Features for Class: {class_name} - {BEST_MODEL_NAME}")
        plt.tight_layout()
        plt.show()
        if shap_sum_list:
        shap_sum = np.abs(shap_sum_list[0]).mean(0)
        for i in range(1, len(shap_sum_list)):
            shap_sum += np.abs(shap_sum_list[i]).mean(0)
            
   shap_genes_df = pd.DataFrame({'mean_abs_shap': shap_sum}, index=feature_names)
        shap_genes_df = shap_genes_df.sort_values('mean_abs_shap', ascending=False)
        top_20_genes = shap_genes_df.head(20).index.tolist()
        print(f"\nTop 20 important genes based on SHAP:\n{top_20_genes}")
    else:
        print("Could not generate top genes list.")
        top_20_genes = []
else:
    print("SHAP values could not be calculated/re-stacked. Skipping plots.")
    top_20_genes = []

print("--- Phase 4 Complete ---")
        
