import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Chargement des données depuis le fichier CSV
cheminFichier = "./vue.csv"
VueDf = pd.read_csv(cheminFichier, sep=";")

print("Colonnes disponibles :", VueDf.columns)

# %% Prétraitement des données
VueDf = VueDf.dropna()  # Suppression des lignes avec valeurs manquantes
VueAr = VueDf.to_numpy()  # Conversion en tableau NumPy (facultatif)

# %% Normalisation des données

# Création d'une colonne contenant la première lettre du prénom
VueDf['initiale'] = VueDf['prenom'].str[0]

# Conversion des lettres en chiffres (A=1, B=2, ..., Z=26)
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
mapping_initiale = {lettre: i + 1 for i, lettre in enumerate(alphabet)}
VueDf['initiale_num'] = VueDf['initiale'].map(mapping_initiale)

# Encodage numérique des mentions au bac
VueDf['mention_bac_num'] = VueDf['mention_bac'].map({
    'P': 1, 'AB': 2, 'B': 3, 'TB': 4
})

# Encodage numérique des niveaux d'étude
VueDf['niveau_etude_num'] = VueDf['niveau_etude'].map({
    "Terminale": 1,
    "Année préparatoire aux études supérieures": 2,
    "1ère année d'études supérieures": 3,
    "2nd année d'études supérieures": 4
})

print(VueDf[['prenom', 'initiale']].head(50))


# %% Diagramme à bâtons : Moyenne en fonction de l'initiale

plt.figure(figsize=(12, 8))
moyennes_par_initiale = VueDf.groupby(
    'initiale')['moyenne'].mean().sort_index()
plt.bar(moyennes_par_initiale.index, moyennes_par_initiale.values,
        color='skyblue', edgecolor='black', linewidth=0.5)
plt.xlabel("Initiale du prénom", fontsize=12)
plt.ylabel("Moyenne", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Afficher les valeurs sur les barres
for i, v in enumerate(moyennes_par_initiale.values):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()

# %% Diagramme à bâtons : Code postal moyen en fonction de l'initiale

plt.figure(figsize=(12, 8))
code_postal_par_initiale = VueDf.groupby(
    'initiale')['code_postal'].mean().sort_index()
plt.bar(code_postal_par_initiale.index, code_postal_par_initiale.values,
        color='orange', edgecolor='black', linewidth=0.5)
plt.xlabel("Initiale du prénom", fontsize=12)
plt.ylabel("Code postal moyen", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Afficher les valeurs sur les barres
for i, v in enumerate(code_postal_par_initiale.values):
    plt.text(i, v + 500, f'{v:.0f}', ha='center', va='bottom', fontsize=10)

plt.show()

# %% Diagramme à bâtons : Niveau d'étude moyen en fonction de l'initiale

plt.figure(figsize=(12, 8))
niveau_etude_par_initiale = VueDf.groupby(
    'initiale')['niveau_etude_num'].mean().sort_index()
plt.bar(niveau_etude_par_initiale.index, niveau_etude_par_initiale.values,
        color='green', edgecolor='black', linewidth=0.5)
plt.xlabel("Initiale du prénom", fontsize=12)
plt.ylabel("Niveau d'étude moyen (numérique)", fontsize=12)
plt.xticks(rotation=0)
plt.yticks([1, 2, 3, 4], ["Terminale", "Prépa", "1ère année", "2ème année"])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Afficher les valeurs sur les barres
for i, v in enumerate(niveau_etude_par_initiale.values):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()

# %% Diagramme à bâtons : Mention au bac moyenne en fonction de l'initiale

plt.figure(figsize=(12, 8))
mention_bac_par_initiale = VueDf.groupby(
    'initiale')['mention_bac_num'].mean().sort_index()
plt.bar(mention_bac_par_initiale.index, mention_bac_par_initiale.values,
        color='purple', edgecolor='black', linewidth=0.5)
plt.xlabel("Initiale du prénom", fontsize=12)
plt.ylabel("Mention au bac moyenne (numérique)", fontsize=12)
plt.xticks(rotation=0)
plt.yticks([1, 2, 3, 4], ["P", "AB", "B", "TB"])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Afficher les valeurs sur les barres
for i, v in enumerate(mention_bac_par_initiale.values):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()

# %% Matrice de corrélation entre les variables numériques

df_corr = VueDf[['initiale_num', 'code_postal',
                 'moyenne', 'mention_bac_num', 'niveau_etude_num']]
corr_matrix = df_corr.corr()
print("Matrice de corrélation :\n", corr_matrix)

# %% Régression linéaire multiple sans sklearn

# Fonctions fournies en tp


def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients de l'hyperplan pour une régression linéaire multiple.
    X : ndarray de shape (n, m)
    y : ndarray de shape (n, 1) ou (n,)
    Retourne : theta (ndarray de shape (m+1,) avec b à l'indice 0)
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta.flatten()


def predire_y(X, theta):
    """
    Calcule y_pred à partir de X et theta.
    X : ndarray de shape (n, m)
    theta : ndarray de shape (m+1,) — inclut l'intercept
    Retourne : y_pred (ndarray de shape (n,))
    """
    n_samples = X.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    y_pred = X_aug @ theta
    return y_pred


def coefficient_correlation_multiple(y_true, y_pred):
    """
    Calcule le coefficient de corrélation multiple (R^2)
    y_true : valeurs réelles (shape: (n,))
    y_pred : valeurs prédites (shape: (n,))
    Retourne : R² (float)
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared

# %% Régression linéaire multiple sans sklearn


X = VueDf[['moyenne', 'mention_bac_num',
           'niveau_etude_num', 'code_postal']].to_numpy()
y = VueDf['initiale_num'].to_numpy()

theta = coefficients_regression_lineaire(X, y)
y_pred = predire_y(X, theta)
r2 = coefficient_correlation_multiple(y, y_pred)

print("Coefficients du modèle :", theta)
print("Coefficient de corrélation multiple R² :", r2)
