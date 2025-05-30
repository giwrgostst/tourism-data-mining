import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

N = 200
ages = np.random.randint(18, 70, N)
income = np.random.randint(1000, 5000, N)
stay_duration = np.random.randint(1, 30, N)
activity_type = np.random.choice(['sea', 'mountain', 'cultural'], N, p=[0.4, 0.3, 0.3])

df = pd.DataFrame({
    'age': ages,
    'income': income,
    'stay_duration': stay_duration,
    'activity_type': activity_type
})

def introduce_missing_values(df, cols, n_missing=5):
    for col in cols:
        df.loc[np.random.choice(df.index, n_missing, replace=False), col] = np.nan

introduce_missing_values(df, ['income', 'stay_duration'])

services_list = ['excursion', 'wine_tour', 'spa', 'museum', 'hiking']
service_data = []
for i in range(N):
    chosen = np.random.choice(services_list, size=np.random.randint(1, 5), replace=False)
    row = {srv: (srv in chosen) for srv in services_list}
    service_data.append(row)
df_services = pd.DataFrame(service_data)

df['income'] = df['income'].fillna(df['income'].median())
df['stay_duration'] = df['stay_duration'].fillna(df['stay_duration'].median())

scaler = MinMaxScaler()
df[['age', 'income', 'stay_duration']] = scaler.fit_transform(df[['age', 'income', 'stay_duration']])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[['age', 'income', 'stay_duration']])
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['age', 'income', 'stay_duration']])

df_pca['cluster'] = df['cluster']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='viridis')
plt.title('Αποτελέσματα Συσταδοποίησης (K-means)')
plt.xlabel('Κύρια Συνιστώσα 1')
plt.ylabel('Κύρια Συνιστώσα 2')
plt.legend(title='Cluster')
plt.show()

frequent_itemsets = apriori(df_services, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, len(frequent_itemsets), metric="confidence", min_threshold=0.5)

print("Κορυφαίοι Κανόνες Συσχέτισης:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())

X_class = df[['age', 'income']]
y_class = df['activity_type']

model_nb = GaussianNB()
model_nb.fit(X_class, y_class)

def min_max_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

scaled_age = min_max_scale(30, 18, 70)
scaled_income = min_max_scale(3000, 1000, 5000)
new_data = pd.DataFrame({'age': [scaled_age], 'income': [scaled_income]})
predicted_activity = model_nb.predict(new_data)
print(f"Προβλεπόμενη προτίμηση δραστηριότητας για νέο τουρίστα: {predicted_activity[0]}")

sns.countplot(x='cluster', data=df)
plt.title('Αριθμός Τουριστών ανά Cluster')
plt.xlabel('Cluster')
plt.ylabel('Πλήθος')
plt.show()

if not rules.empty:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', data=rules)
    plt.title('Μετρικές Κανόνων Συσχέτισης')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.show()