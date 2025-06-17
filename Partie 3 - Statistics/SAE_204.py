import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers le fichier CSV
cheminFichier = "C:/Users/dougl/Mon Drive (douglasse.ylc@gmail.com)/BUT_1E2/Semestre-2/SAE-2.04/Partie 3 - Analyse statistique des données/vue_voyelles.csv"

# Correction ici : ajout de sep=";" car le fichier utilise des points-virgules
VueDf = pd.read_csv(cheminFichier, sep=";")

print("Colonnes disponibles :", VueDf.columns)


# === Conversion en tableau numpy (facultatif) ===
Vue = VueDf.to_numpy()
VueStr = Vue[:, :1].astype(str)
VueNum = Vue[:, 1:].astype(float)

# === Fonction pour extraire la première voyelle d'un prénom ===


def premiere_voyelle(prenom):
    voyelles = "aeiouyAEIOUY"
    for lettre in prenom:
        if lettre in voyelles:
            return lettre.lower()
    return None


# === Application de la fonction sur la colonne des prénoms ===
VueDf['voyelle_prenom'] = VueDf['prenom'].apply(premiere_voyelle)

# === Affichage du nombre de prénoms par voyelle ===
print("\nNombre de prénoms par voyelle :")
print(VueDf['voyelle_prenom'].value_counts())

# === Création des colonnes binaires pour chaque voyelle ===
voyelle_dummies = pd.get_dummies(VueDf['voyelle_prenom'], prefix="voyelle")
VueDf = pd.concat([VueDf, voyelle_dummies], axis=1)
