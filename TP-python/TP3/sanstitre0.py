import numpy as np
import pandas as pd
# %%

chemin_fichier = "C:/Users/dougl/Mon Drive (douglasse.ylc@gmail.com)/BUT_1E2/Semestre-2/SAÉ-2.04/TP-python/TP3/Vue.csv"

vue_df = pd.read_csv(chemin_fichier)
print(vue_df.columns)

vue_str = vue_df[['ine', 'cat_socio_parent', 'serie_bac']].to_numpy(type(str))
print(vue_str)

vue_num = vue_df[['classement', 'm1101', 'm1102',
                  'm1103', 'm1104', 'm1105', 'm1106', 'm2101', 'm2102', 'm2103', 'm2104',
                  'm2105', 'm2106', 'm2107', 'UE11', 'UE12', 'UE21', 'UE22']].to_numpy(type(float()))
print(vue_num)

# %%
correls = np.corrcoef(vue_num, rowvar=False)

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


theta = coefficients_regression_lineaire(X, Y)
print("theta")

# %%
