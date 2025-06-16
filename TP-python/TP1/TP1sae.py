import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%   1) Récupération des données


df = pd.read_csv(
    "C:/Users/dougl/Mon Drive (douglasse.ylc@gmail.com)/BUT_1E2/Semestre-2/R2.08- Statistiques/SAE04/TP1/Sangliers.csv")
print(df.head())
nb_sanglier_preleves = df['Nb_sanglier_preleves'].to_numpy()
nb_permis_chasse = df['Nb_permis_chasse'].to_numpy()

# %%   2a) Fonction moyenne


def moyenne(tab):
    return sum(tab) / len(tab)


print("Moyenne :", moyenne(nb_sanglier_preleves))
print("Moyenne :", np.mean(nb_sanglier_preleves))

# %%   2b) Fonction variance


def variance(tab):
    m = moyenne(tab)
    return sum((x - m) ** 2 for x in tab) / len(tab)


print("Variance :", variance(nb_sanglier_preleves))
print("Variance :", np.var(nb_sanglier_preleves, ddof=0))

# %%   2c) Fonction écart-type


def ecart_type(tab):
    return np.sqrt(variance(tab))


print("Écart-type :", ecart_type(nb_sanglier_preleves))
print("Écart-type :", np.std(nb_sanglier_preleves, ddof=0))

# %%   2d) Fonction covariance


def covariance(X, Y):
    m1 = moyenne(X)
    m2 = moyenne(Y)
    return sum((X[i] - m1) * (Y[i] - m2) for i in range(len(X))) / len(Y)


print("Covariance :", covariance(nb_sanglier_preleves, nb_permis_chasse))
print("Covariance :", np.cov(
    nb_sanglier_preleves, nb_permis_chasse, ddof=0)[0, 1])

# %%   2e) Fonction corrélation


def correlation(X, Y):
    return covariance(X, Y) / (ecart_type(X) * ecart_type(Y))


print("Corrélation :", correlation(nb_sanglier_preleves, nb_permis_chasse))
print("Corrélation :", np.corrcoef(
    nb_sanglier_preleves, nb_permis_chasse)[0, 1])


# %%


def norm(X):
