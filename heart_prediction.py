# importation des bibliotheque
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


import pandas as pd
#****Téléchargement et Affichage des Premières Lignes d'un Fichier CSV depuis Google Drive en Utilisant Pandas****#

# Lien direct vers dataset dans drive
url = 'https://drive.google.com/uc?id=1qv9rpuygu1LpRmxI29tHVdJUH64-ggci'
# Télécharger et lire le fichier CSV
df = pd.read_csv(url)

# Afficher les premières lignes
print(df.head())

#****Génération de Statistiques Descriptives pour un DataFrame en Utilisant Pandas***#

df.describe()
#****Calcul du Nombre de Valeurs Manquantes dans Chaque Colonne d'un DataFrame en Utilisant Pandas****#
df.isnull().sum()

#****Visualisation de la Distribution de l'Âge avec un Histogramme et une Estimation de la Densité Kernel "EDK" en Utilisant Seaborn***#

# Distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#*****Diagrammes en Barres des Caractéristiques Binaires en Utilisant Seaborn dans une Mise en Page 2x3***#

# Count plot of binary features
binary_features = ['anaemia', 'high_blood_pressure', 'diabetes', 'sex', 'smoking', 'DEATH_EVENT']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(binary_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=df, x=feature, palette='Set2')
    plt.title(f'Count Plot of {feature}')
    plt.xlabel('')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


#*****Diagramme de Chaleur de Corrélation des Caractéristiques Numériques avec Annotations en Utilisant Seaborn***#

# Correlation heatmap of numeric features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


#*****Entraînement d'un Classificateur RandomForest sur un Jeu de Données Fractionné en Utilisant Scikit-Learn*****#

# Machine learning approach: Random Forest Classifier
# Splitting the data into features (X) and target variable (y)
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)



#****Hyperparametre****
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200], # Nombre d'arbres dans la forêt
    'max_depth': [None, 10, 20, 30], # Profondeur maximale de chaque arbre
    'min_samples_split': [2, 5, 10], # Nombre minimum d'échantillons requis pour diviser un nœud
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:\n", grid_search.best_params_)

# Train the model with the best hyperparameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

#***Évaluation d'un Classificateur RandomForest : Précision, Rapport de Classification et Visualisation de la Matrice de Confusion***#

# Model evaluation
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#***SVC*
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Initialize SVM classifier
# Model training
svc_classifier = SVC(random_state=42)
svc_classifier.fit(X_train, y_train)  # Fit the model
# Predictions
y_predSV = svc_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predSV))

#*****KNN****

from sklearn.neighbors import KNeighborsClassifier
# Créer un modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Vous pouvez ajuster le nombre de voisins (n_neighbors) en fonction de votre problème
# Entraîner le modèle sur l'ensemble d'entraînement
knn_model.fit(X_train, y_train)
# Faire des prédictions sur l'ensemble de test
y_predKNN = knn_model.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
# Classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


#********Meilleur
from sklearn.model_selection import cross_val_score
# Initialize models
models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC()
}

# Perform cross-validation and evaluate each model
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print(f"***{model_name}:==> Mean Accuracy: {scores.mean()},==> Std: {scores.std()}")

# Choose the best performing model
best_model_name = max(models, key=lambda k: scores.mean())
print(f"Best Model: {best_model_name}")

from sklearn.tree import expert_graphviz
import pydotplus
from IPython.display import image

#Extract one of the decision trees
tree = best_rf.estimators_[0]

#Export the decision tree to a DOT file
dot_data = expert_graphviz(tree, out_file=None,
                           feature_name=X_train.columns, #Assuming X_train is a DataFrame
                           class_names=['class_0','class_1'],
                           filled=True, rounded=True,
                           special_characters=True)

#Create a Graphviz object from the Dot data
graph = pydotplus.graph_from_dot_data(dot_data)

#Display the decision tree using Graphviz
image(graph.create_png())