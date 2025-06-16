import numpy as np
import pandas as pd
# %%
chemin_file = "C:/Users/dougl/Mon Drive (douglasse.ylc@gmail.com)/BUT_1E2/Semestre-2/R2.08- Statistiques/SAE04/TP1/Sangliers.csv"

sa_df = pd.read_csv(chemin_file)
print(sa_df.head())

sa_ar = sa_df.to_numpy(dtype=float)

# %%

print(sa_ar)
sa_ar[:, 1] = (sa_ar[:, 1]-np.mean(sa_ar[:, 1]))/np.std(sa_ar[:, 1], ddof=0)
print(np.mean(sa_ar[:, 1])), np.std(sa_ar[:, 1], ddof=0)


for i in range(1, 5):
    sa_ar[:, 1] = (sa_ar[:, 1]-np.mean(sa_ar[:, 1])) / \
        np.std(sa_ar[:, 1], ddof=0)
    print(np.mean(sa_ar[:, 1])), np.std(sa_ar[:, 1], ddof=0)

X = np.concatenate(self, (a1, a2, ...))te((sa_ar[:, 1:2], sa_ar[:, 3:5]), axis=1)
Y = sa_ar[:, 2]
# %%


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


# %%
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


# %%


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
