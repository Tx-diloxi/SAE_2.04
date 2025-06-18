import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Chargement des données depuis le fichier CSV
cheminFichier = "C:/Users/dougl/Mon Drive (douglasse.ylc@gmail.com)/BUT_1E2/Semestre-2/SAE-2.04/Partie 3 - Analyse statistique des données/vue.csv"
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

# %% Diagramme : Nombre d’étudiants par moyenne (arrondie)

VueDf['moyenne_arrondie'] = VueDf['moyenne'].round().astype(int)
moyenne_counts = VueDf['moyenne_arrondie'].value_counts().sort_index()

x = list(range(0, 21))
y = [moyenne_counts.get(i, 0) for i in x]

plt.figure(figsize=(12, 6))
plt.bar(x, y, color='slateblue')
plt.title("Nombre d'étudiants par moyenne (arrondie)")
plt.xlabel("Moyenne")
plt.ylabel("Nombre d'étudiants")
plt.xticks(x)
plt.grid(axis='y', alpha=0.8)
plt.show()

# %% Diagramme : Nombre d’étudiants par mention au bac

mentions = ['P', 'AB', 'B', 'TB']
frequences_mentions = [
    VueDf[VueDf['mention_bac'] == m].shape[0] for m in mentions]

plt.figure(figsize=(8, 5))
plt.bar(mentions, frequences_mentions, color='mediumseagreen')
plt.title("Répartition des étudiant·es selon la mention au bac")
plt.xlabel("Mention")
plt.ylabel("Nombre d'étudiant·es")
plt.grid(axis='y', alpha=0.8)
plt.show()

# %% Diagramme : Moyenne par initiale de prénom

initiales = sorted(VueDf['initiale'].unique())
moyennes_initiales = [VueDf[VueDf['initiale'] == ini]
                      ['moyenne'].mean() for ini in initiales]

plt.figure(figsize=(12, 6))
plt.bar(initiales, moyennes_initiales, color='orange')
plt.title("Moyenne par initiale de prénom")
plt.xlabel("Initiale")
plt.ylabel("Moyenne")
plt.grid(axis='y', alpha=0.7)
plt.show()

# %% Diagramme : Répartition des étudiant·es selon le niveau d’étude

niveaux = VueDf['niveau_etude'].unique()
frequences_niveaux = [VueDf[VueDf['niveau_etude'] == n].shape[0]
                      for n in niveaux]

plt.figure(figsize=(10, 6))
plt.bar(niveaux, frequences_niveaux, color='salmon')
plt.title("Répartition des étudiant·es selon le niveau d’étude")
plt.xlabel("Niveau d’étude")
plt.ylabel("Nombre d’étudiant·es")
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', alpha=0.8)
plt.tight_layout()
plt.show()

# %% Diagramme : Moyenne par code postal (regroupé par centaine)

VueDf['code_postal_groupe'] = VueDf['code_postal'] // 100
groupes_cp = sorted(VueDf['code_postal_groupe'].unique())
moyennes_cp = [VueDf[VueDf['code_postal_groupe'] == g]['moyenne'].mean()
               for g in groupes_cp]

plt.figure(figsize=(12, 6))
plt.bar(groupes_cp, moyennes_cp, color='blue')
plt.title("Moyenne par code postal (regroupé par centaine)")
plt.xlabel("Code postal (x100)")
plt.ylabel("Moyenne")
plt.grid(axis='y', alpha=1)
plt.show()

# %% Matrice de corrélation entre les variables numériques

df_corr = VueDf[['initiale_num', 'code_postal',
                 'moyenne', 'mention_bac_num', 'niveau_etude_num']]
corr_matrix = df_corr.corr()
print("Matrice de corrélation :\n", corr_matrix)

# %% Régression linéaire multiple sans sklearn

# --- Fonctions mathématiques pour la régression ---


def coefficients_regression_lineaire(X, y):
    """
    Calcule les coefficients de l'hyperplan pour une régression linéaire multiple.

    X : ndarray de shape (n, m)
    y : ndarray de shape (n, 1) ou (n,)

    Retourne : theta (ndarray de shape (m+1,) avec b à l'indice 0)
    """
    n_samples = X.shape[0]

    # Ajouter une colonne de 1 pour l'ordonnée à l'origine
    X_aug = np.hstack((np.ones((n_samples, 1)), X))  # shape: (n, m+1)

    # Formule des moindres carrés
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
    X_aug = np.hstack((np.ones((n_samples, 1)), X))  # ajoute une colonne de 1
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



# --- Régression avec pour cible l’ordre alphabétique des prénoms ---
X = VueDf[['moyenne', 'mention_bac_num',
           'niveau_etude_num', 'code_postal']].to_numpy()
y = VueDf['initiale_num'].to_numpy()

theta = coefficients_regression_lineaire(X, y)
y_pred = predire_y(X, theta)
r2 = coefficient_correlation_multiple(y, y_pred)

# --- Résultats ---
print("Coefficients du modèle (theta) :", theta)
print("Coefficient de corrélation multiple R² :", r2)
