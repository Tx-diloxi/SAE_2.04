import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Import et préparation des données
chemin_fichier = "vue_voyelles_m2105.csv"

# Import des données
vue_df = pd.read_csv(chemin_fichier, sep=';')
print("Colonnes disponibles:", vue_df.columns)

# Sélection des colonnes catégorielles
vue_str = vue_df[['prenom', 'code_nip']].to_numpy(dtype=str)
print("Données catégorielles:\n", vue_str[:5])

# Sélection des colonnes numériques
vue_num = vue_df[['m2105', 'm1101', 'm1102', 'm1201', 'm2101']].to_numpy(dtype=float)
print("Données numériques:\n", vue_num[:5])

# %% Visualisation des distributions
plt.figure(figsize=(15, 5))

# Histogramme des notes m2105
plt.subplot(131)
plt.hist(vue_num[:, 0], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution des notes m2105')
plt.xlabel('Notes')
plt.ylabel('Fréquence')

# Histogramme des notes m1101
plt.subplot(132)
plt.hist(vue_num[:, 1], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution des notes m1101')
plt.xlabel('Notes')

# Histogramme des notes m1102
plt.subplot(133)
plt.hist(vue_num[:, 2], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution des notes m1102')
plt.xlabel('Notes')

plt.tight_layout()
plt.show()

# %% Normalisation des données
vue_norm = (vue_num - np.mean(vue_num, axis=0)) / np.std(vue_num, axis=0)
print("\nDonnées normalisées (5 premières lignes):")
print(vue_norm[:5])

# %% Calcul et visualisation de la matrice de corrélation
correls = np.corrcoef(vue_num.T)
print("\nMatrice de corrélation:")
print(correls)

plt.figure(figsize=(10, 8))
plt.imshow(correls, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(['m2105', 'm1101', 'm1102', 'm1201', 'm2101'])), 
           ['m2105', 'm1101', 'm1102', 'm1201', 'm2101'], rotation=45)
plt.yticks(range(len(['m2105', 'm1101', 'm1102', 'm1201', 'm2101'])), 
           ['m2105', 'm1101', 'm1102', 'm1201', 'm2101'])
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.show()

# %% Fonctions de régression linéaire
def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients de l'hyperplan pour une régression linéaire multiple.
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta.flatten()

def predire_y(X, theta):
    """
    Calcule y_pred à partir de X et theta.
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    y_pred = X_aug @ theta
    return y_pred

def coefficient_correlation_multiple(y_true, y_pred):
    """
    Calcule le coefficient de corrélation multiple (R^2)
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared

# %% Application de la régression linéaire
# Prédiction de m2105 à partir des autres variables
X = vue_norm[:, 1:]  # Toutes les colonnes sauf m2105
y = vue_norm[:, 0]   # m2105

# Calcul des coefficients
theta = coefficients_regression_lineaire(X, y)
print("\nCoefficients de régression:", theta)

# Prédictions
y_pred = predire_y(X, theta)

# Calcul du R²
r2 = coefficient_correlation_multiple(y, y_pred)
print("R² =", r2)

# %% Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Notes réelles (normalisées)')
plt.ylabel('Notes prédites (normalisées)')
plt.title('Prédictions vs Réalité - m2105')
plt.show()