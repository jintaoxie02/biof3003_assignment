import json
import nbformat

def update_notebook():
    # Read the notebook
    with open('ppg_signal_processing.ipynb', 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Update imports cell
    imports_cell = notebook['cells'][2]
    imports_cell['source'] = """# Importing libraries
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import find_peaks
import seaborn as sns

# Checking versions
print("Library versions:")
print(f"pymongo version: {pymongo.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")"""

    # Update MongoDB connection cell
    mongo_cell = notebook['cells'][4]
    mongo_cell['source'] = """# Connecting to MongoDB Atlas
client = pymongo.MongoClient("mongodb+srv://biof3003digitalhealth01:qoB38jemj4U5E7ZL@cluster0.usbry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["test"]
collection = db["records"]

# Getting all the data
data = list(collection.find({}))
df = pd.DataFrame(data)

print(f"Successfully connected to MongoDB Atlas!")
print(f"Found {len(df)} records in the database")
print("\\nColumns in the data:")
print(df.columns.tolist())
print("\\nSample record:")
print(df[['heartRate', 'confidence', 'timestamp']].head())"""

    # Update plotting function cell
    plot_cell = notebook['cells'][6]
    plot_cell['source'] = """def plot_ppg_signal(signal, title="PPG Signal"):
    \"\"\"Plot a single PPG signal with peaks highlighted.\"\"\"
    plt.figure(figsize=(15, 5))
    plt.plot(signal, label='Signal')
    
    # Find peaks in the signal
    peaks, _ = find_peaks(signal, distance=30)  # Adjust distance based on your sampling rate
    plt.plot(peaks, [signal[i] for i in peaks], "x", label='Peaks')
    
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return peaks

# Plot a few sample signals
for i in range(3):
    signal = df.iloc[i]['ppgData']
    peaks = plot_ppg_signal(signal, f"PPG Signal {i+1}")
    print(f"Signal {i+1} - Number of peaks: {len(peaks)}")"""

    # Update feature extraction cell
    features_cell = notebook['cells'][8]
    features_cell['source'] = """def extract_features(signal):
    \"\"\"Extract time and frequency domain features from a PPG signal.\"\"\"
    features = {}
    
    # Check if signal is empty or invalid
    if not isinstance(signal, (list, np.ndarray)) or len(signal) == 0:
        # Return default values for invalid signals
        features['mean'] = np.nan
        features['std'] = np.nan
        features['max'] = np.nan
        features['min'] = np.nan
        features['range'] = np.nan
        features['rms'] = np.nan
        features['num_peaks'] = np.nan
        features['mean_peak_interval'] = np.nan
        features['std_peak_interval'] = np.nan
        features['dominant_freq'] = np.nan
        features['total_power'] = np.nan
        return features
    
    try:
        # Convert to numpy array if it's a list
        signal = np.array(signal, dtype=float)
        
        # Time domain features
        features['mean'] = float(np.nanmean(signal))
        features['std'] = float(np.nanstd(signal))
        features['max'] = float(np.nanmax(signal))
        features['min'] = float(np.nanmin(signal))
        features['range'] = float(features['max'] - features['min'])
        features['rms'] = float(np.sqrt(np.nanmean(np.square(signal))))
        
        # Peak detection features
        peaks, _ = find_peaks(signal, distance=30)
        features['num_peaks'] = int(len(peaks))
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            features['mean_peak_interval'] = float(np.nanmean(peak_intervals))
            features['std_peak_interval'] = float(np.nanstd(peak_intervals))
        else:
            features['mean_peak_interval'] = np.nan
            features['std_peak_interval'] = np.nan
        
        # Frequency domain features
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal))
        power = np.abs(fft)**2
        
        features['dominant_freq'] = float(abs(freq[np.argmax(power)]))
        features['total_power'] = float(np.nansum(power))
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Return NaN for all features if there's an error
        for key in features:
            features[key] = np.nan
    
    return features

# Extract features from all signals
print("Extracting features from all signals...")
features_list = []
for idx, row in df.iterrows():
    try:
        # Ensure ppgData is a list/array
        ppg_data = row['ppgData']
        if isinstance(ppg_data, dict):
            print(f"Warning: ppgData at index {idx} is a dictionary, skipping...")
            continue
            
        features = extract_features(ppg_data)
        
        # Handle confidence value
        confidence = row['confidence']
        if isinstance(confidence, dict):
            confidence = confidence.get('value', np.nan)
        features['confidence'] = float(confidence)
        
        # Handle heart rate value
        heart_rate = row['heartRate']
        if isinstance(heart_rate, dict):
            heart_rate = heart_rate.get('value', np.nan)
        features['heart_rate'] = float(heart_rate)
        
        features_list.append(features)
    except Exception as e:
        print(f"Error processing signal at index {idx}: {str(e)}")
        continue

# Create features DataFrame
features_df = pd.DataFrame(features_list)

# Clean the data
print("\\nCleaning data...")
# Drop rows where all features are NaN
features_df = features_df.dropna(how='all')
# Drop rows where confidence is NaN
features_df = features_df.dropna(subset=['confidence'])
# Fill remaining NaN values with column means
features_df = features_df.fillna(features_df.mean())

# Define signal quality based on confidence score
features_df['quality'] = pd.cut(features_df['confidence'],
                               bins=[0, 60, 80, 100],
                               labels=['poor', 'acceptable', 'good'])

# Ensure all quality labels are valid
features_df = features_df.dropna(subset=['quality'])

print("\\nFeatures extracted:")
print(features_df.columns.tolist())
print("\\nSample of extracted features:")
print(features_df.head())
print("\\nNumber of records after cleaning:", len(features_df))
print("\\nQuality distribution:")
print(features_df['quality'].value_counts())"""

    # Update model training cell
    model_cell = notebook['cells'][10]
    model_cell['source'] = """# Prepare features and target
X = features_df.drop(['quality', 'confidence'], axis=1)  # Remove quality and confidence from features
y = features_df['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()"""

    # Update model evaluation cell
    eval_cell = notebook['cells'][11]
    eval_cell['source'] = """# Plot feature distributions by quality class
plt.figure(figsize=(15, 10))
for i, feature in enumerate(X.columns):
    plt.subplot(4, 4, i+1)
    for quality in ['poor', 'acceptable', 'good']:
        sns.kdeplot(data=features_df[features_df['quality'] == quality][feature], 
                   label=quality, fill=True)
    plt.title(feature)
    plt.legend()
plt.tight_layout()
plt.show()

# Plot correlation matrix (excluding categorical columns)
plt.figure(figsize=(12, 8))
numeric_features = features_df.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Plot confusion matrix with percentages
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues')
plt.title('Confusion Matrix (Percentages)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()"""

    # Save the updated notebook
    with open('ppg_signal_processing.ipynb', 'w') as f:
        nbformat.write(notebook, f)

if __name__ == "__main__":
    update_notebook() 