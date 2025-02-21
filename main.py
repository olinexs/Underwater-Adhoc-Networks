import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'path/underwater_network_data_v3.csv' with your actual path)
df = pd.read_csv("path/underwater_network_data_v3.csv")

print(df.info())
print(df.describe())

# Data Preprocessing

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in df.columns:
    if df[col].dtype != 'object': 
        df = remove_outliers(df, col)

plt.figure(figsize=(7,5))
sns.scatterplot(x=df["Transmission Power (dB)"], y=df["Throughput (kbps)"], color="blue")
plt.title("Transmission Power vs. Throughput")
plt.xlabel("Transmission Power (dB)")
plt.ylabel("Throughput (kbps)")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x=df["Signal Attenuation (dB/m)"], y=df["Packet Delivery Ratio (PDR, %)"], color="red")
plt.title("Signal Attenuation vs. PDR")
plt.xlabel("Signal Attenuation (dB/m)")
plt.ylabel("Packet Delivery Ratio (PDR, %)")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df["Latency (ms)"], bins=30, kde=True, color="green")
plt.title("Latency Distribution")
plt.xlabel("Latency (ms)")
plt.show()

df_numeric = df.drop(columns=["Node ID", "Neighbors"]) 

# Convert data to float (in case some values were read as strings)
df_numeric = df_numeric.apply(pd.to_numeric, errors="coerce")

plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Underwater Network Data")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

features = df_numeric.drop(columns=["Throughput (kbps)", "Latency (ms)", "Packet Delivery Ratio (PDR, %)", "Energy Consumption (J)"])
targets = df_numeric[["Throughput (kbps)", "Latency (ms)", "Packet Delivery Ratio (PDR, %)", "Energy Consumption (J)"]]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {}
predictions = {}
metrics = {}

for col in targets.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train[col])
    y_pred = model.predict(X_test_scaled)
    
    models[col] = model
    predictions[col] = y_pred

    metrics[col] = {
        "MAE": mean_absolute_error(y_test[col], y_pred),
        "MSE": mean_squared_error(y_test[col], y_pred),
        "R2 Score": r2_score(y_test[col], y_pred)
    }

metrics


features_to_plot = ["Throughput (kbps)", "Latency (ms)", "Packet Delivery Ratio (PDR, %)", "Mobility (m/s)"]
plt.figure(figsize=(12, 8))

for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], bins=30, kde=True, color="blue")
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()

import numpy as np

# Check unique values in PDR and Energy Consumption to verify constraints
pdr_unique = df_numeric["Packet Delivery Ratio (PDR, %)"].unique()
energy_unique = df_numeric["Energy Consumption (J)"].unique()

# Compute basic statistics to check for anomalies
pdr_stats = df_numeric["Packet Delivery Ratio (PDR, %)"].describe()
energy_stats = df_numeric["Energy Consumption (J)"].describe()

pdr_unique, energy_unique, pdr_stats, energy_stats


# %%
# Compute correlation matrix to analyze relationships
correlation_matrix = df_numeric.corr()

# Extract correlations of PDR and Energy Consumption with other variables
pdr_correlations = correlation_matrix["Packet Delivery Ratio (PDR, %)"].sort_values(ascending=False)
energy_correlations = correlation_matrix["Energy Consumption (J)"].sort_values(ascending=False)

pdr_correlations, energy_correlations


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set_style("whitegrid")

# Create subplots for PDR and Energy Consumption distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot PDR distribution
sns.histplot(df["Packet Delivery Ratio (PDR, %)"], bins=20, kde=True, ax=axes[0], color="blue")
axes[0].set_title("Distribution of Packet Delivery Ratio (PDR)")

# Plot Energy Consumption distribution
sns.histplot(df["Energy Consumption (J)"], bins=20, kde=True, ax=axes[1], color="red")
axes[1].set_title("Distribution of Energy Consumption (J)")

plt.show()


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = df_numeric.drop(columns=["Energy Consumption (J)"])

features = df_numeric.drop(columns=["Throughput (kbps)", "Latency (ms)", "Packet Delivery Ratio (PDR, %)"])
targets = df_numeric[["Throughput (kbps)", "Latency (ms)", "Packet Delivery Ratio (PDR, %)"]]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R² Score": r2_score(y_test, y_pred)
    }
    
    results[name] = metrics

import pprint
pprint.pprint(results)


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the best model (XGBoost)
xgb_feature_importance = models["XGBoost"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": features.columns,
    "Importance": xgb_feature_importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.title("Feature Importance in XGBoost Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()



# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define efficiency criteria based on dataset statistics
throughput_threshold = df_numeric["Throughput (kbps)"].median()
pdr_threshold = df_numeric["Packet Delivery Ratio (PDR, %)"].median()
latency_threshold = df_numeric["Latency (ms)"].median()

# Label data as Efficient (1) or Inefficient (0)
df_numeric["Efficiency_Label"] = ((df_numeric["Throughput (kbps)"] > throughput_threshold) & 
                          (df_numeric["Packet Delivery Ratio (PDR, %)"] > pdr_threshold) & 
                          (df_numeric["Latency (ms)"] < latency_threshold)).astype(int)

# Select features for classification
classification_features = ["Communication Range (m)", "Transmission Power (dB)", "Noise Level (dB)", 
                           "Temperature (°C)", "Salinity (PSU)"]

X = df_numeric[classification_features]
y = df_numeric["Efficiency_Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)



# %%
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Select key performance metrics for clustering
clustering_features = df_numeric[["Throughput (kbps)", "Packet Delivery Ratio (PDR, %)", "Latency (ms)"]]

# Standardize data
scaler = StandardScaler()
clustering_features_scaled = scaler.fit_transform(clustering_features)

# Apply K-Means clustering (using 3 clusters: Low, Medium, High performance)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_numeric["Cluster"] = kmeans.fit_predict(clustering_features_scaled)

# Visualize Clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x=df_numeric["Throughput (kbps)"], y=df_numeric["Latency (ms)"], hue=df_numeric["Cluster"], palette="viridis")
plt.title("Clustering of Network Nodes by Performance")
plt.xlabel("Throughput (kbps)")
plt.ylabel("Latency (ms)")
plt.legend(title="Cluster")
plt.show()

# Print cluster statistics
df_numeric.groupby("Cluster")[["Throughput (kbps)", "Packet Delivery Ratio (PDR, %)", "Latency (ms)"]].mean()



# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Extract necessary features
X = df_numeric['X']
Y = df_numeric['Y']
Z = df_numeric['Z']
throughput = df_numeric['Throughput (kbps)']
latency = df_numeric['Latency (ms)']
cluster_labels = df_numeric['Cluster'] if 'Cluster' in df_numeric.columns else None

# Normalize throughput for coloring
scaler = MinMaxScaler()
thr_scaled = scaler.fit_transform(throughput.values.reshape(-1, 1)).flatten()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes with color based on throughput
sc = ax.scatter(X, Y, Z, c=thr_scaled, cmap='coolwarm', s=50, alpha=0.8)

# Color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Normalized Throughput")

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Network Visualization Based on Throughput")

plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Extract necessary features
X = df_numeric['X']
Y = df_numeric['Y']
Z = df_numeric['Z']
throughput = df_numeric['Throughput (kbps)']
latency = df_numeric['Latency (ms)']
cluster_labels = df_numeric['Cluster'] if 'Cluster' in df_numeric.columns else None

# Normalize throughput for coloring
scaler = MinMaxScaler()
thr_scaled = scaler.fit_transform(throughput.values.reshape(-1, 1)).flatten()

# Define communication range (adjustable parameter)
COMM_RANGE = 50

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes with color based on throughput
sc = ax.scatter(X, Y, Z, c=thr_scaled, cmap='coolwarm', s=50, alpha=0.8, edgecolors='k')

# Add edges based on distance threshold
for i in range(len(df_numeric)):
    for j in range(i + 1, len(df_numeric)):
        dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
        if dist < COMM_RANGE:
            ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], c='black', alpha=0.3, linewidth=0.8)

# Color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Normalized Throughput")

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Network Visualization with Connectivity")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Extract necessary features
X = df_numeric['X']
Y = df_numeric['Y']
Z = df_numeric['Z']
throughput = df_numeric['Throughput (kbps)']
latency = df_numeric['Latency (ms)']
cluster_labels = df_numeric['Cluster'] if 'Cluster' in df_numeric.columns else None

scaler = MinMaxScaler()
thr_scaled = scaler.fit_transform(throughput.values.reshape(-1, 1)).flatten()

# Define communication range
COMM_RANGE = 150

# Select important nodes: highest throughput or clustered
important_threshold = np.percentile(throughput, 85)  # Top 15% throughput
important_nodes = throughput >= important_threshold

# Filtered node coordinates
X_imp, Y_imp, Z_imp = X[important_nodes], Y[important_nodes], Z[important_nodes]
thr_scaled_imp = thr_scaled[important_nodes]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot only important nodes
sc = ax.scatter(X_imp, Y_imp, Z_imp, c=thr_scaled_imp, cmap='coolwarm', s=100, alpha=0.9, edgecolors='k')

# Add edges only for important nodes
for i in range(len(X_imp)):
    for j in range(i + 1, len(X_imp)):
        dist = np.sqrt((X_imp.iloc[i] - X_imp.iloc[j])**2 + (Y_imp.iloc[i] - Y_imp.iloc[j])**2 + (Z_imp.iloc[i] - Z_imp.iloc[j])**2)
        if dist < COMM_RANGE:
            ax.plot([X_imp.iloc[i], X_imp.iloc[j]], [Y_imp.iloc[i], Y_imp.iloc[j]], [Z_imp.iloc[i], Z_imp.iloc[j]], 
                    c='black', alpha=0.6, linewidth=1.2)

cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label("Normalized Throughput")

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Filtered 3D Network Visualization with Key Nodes and Connectivity")

plt.show()


x, y, z = df_numeric['X'], df_numeric['Y'], df_numeric['Z']
throughput = df_numeric['Throughput (kbps)']

# Filter nodes based on throughput threshold (only show stronger nodes)
threshold = np.percentile(throughput, 85)
strong_nodes = df_numeric[throughput >= threshold]

G = nx.Graph()

for i, row in strong_nodes.iterrows():
    G.add_node(i, pos=(row['X'], row['Y'], row['Z']), throughput=row['Throughput (kbps)'])

# Add edges - only to the 3 nearest stronger neighbors
for i, row in strong_nodes.iterrows():
    distances = np.linalg.norm(strong_nodes[['X', 'Y', 'Z']].values - np.array([row['X'], row['Y'], row['Z']]), axis=1)
    nearest_neighbors = np.argsort(distances)[1:4] 
    for neighbor in nearest_neighbors:
        G.add_edge(i, strong_nodes.index[neighbor])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

pos = nx.get_node_attributes(G, 'pos')
colors = [nx.get_node_attributes(G, 'throughput')[n] for n in G.nodes()]

scatter = ax.scatter(*zip(*pos.values()), c=colors, cmap='coolwarm', edgecolors='black', s=80)

for edge in G.edges():
    x_vals, y_vals, z_vals = zip(*[pos[edge[0]], pos[edge[1]]])
    ax.plot(x_vals, y_vals, z_vals, color='black', linewidth=0.8, alpha=0.7)

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Key Signal Propagation & Attenuation')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label('Normalized Throughput')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))

df_numeric["Packet Delivery Ratio (PDR, %)"] = pd.to_numeric(df_numeric["Packet Delivery Ratio (PDR, %)"], errors="coerce")

sns.scatterplot(x=df_numeric["Latency (ms)"], y=df_numeric["Throughput (kbps)"], hue=df_numeric["Packet Delivery Ratio (PDR, %)"], 
                palette="coolwarm", edgecolor="black")

plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (kbps)")
plt.title("Throughput vs. Latency Scatter Plot")
plt.legend(title="PDR (%)")
plt.grid(True)
plt.show()


num_bins = 20 
df_numeric['X_bin'] = pd.cut(df_numeric['X'], bins=num_bins, labels=False)
df_numeric['Y_bin'] = pd.cut(df_numeric['Y'], bins=num_bins, labels=False)

heatmap_data = df_numeric.pivot_table(index='X_bin', columns='Y_bin', values='Packet Delivery Ratio (PDR, %)', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="viridis", cbar=True)

plt.title("Heatmap of Packet Delivery Ratio")
plt.xlabel("Y Coordinate (Binned)")
plt.ylabel("X Coordinate (Binned)")
plt.show()


heatmap_data = df_numeric.pivot_table(index='X_bin', columns='Y_bin', values='Throughput (kbps)', aggfunc='mean')

plt.figure(figsize=(10, 8))

sns.heatmap(heatmap_data, cmap='viridis', annot=False, linewidths=0.5)

plt.title("Heatmap of Network Throughput")
plt.xlabel("Y Coordinate (Binned)")
plt.ylabel("X Coordinate (Binned)")

plt.show()

heatmap_data_latency = df_numeric.pivot_table(index='X_bin', columns='Y_bin', values='Latency (ms)', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_latency, cmap='viridis', annot=False, linewidths=0.5)

plt.title("Heatmap of Network Latency")
plt.xlabel("Y Coordinate (Binned)")
plt.ylabel("X Coordinate (Binned)")

plt.show()


