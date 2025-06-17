import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Import et préparation des données
chemin_fichier = "vue_voyelles_m2105.csv"
matieres = ['m2105', 'm1101', 'm1102', 'm1201', 'm2101']

# Import des données
vue_df = pd.read_csv(chemin_fichier, sep=';')

# %% Fonction pour extraire les voyelles
def compter_voyelles(prenom):
    """Compte les voyelles dans un prénom"""
    voyelles = 'aeiouAEIOU'
    return ''.join(c for c in prenom if c in voyelles).lower()

# Ajouter la colonne des voyelles
vue_df['voyelles'] = vue_df['prenom'].apply(compter_voyelles)

# %% Analyse pour chaque matière
def analyser_matiere(df, matiere):
    """Analyse la relation entre voyelles et notes pour une matière"""
    # Calculer moyennes par combinaison de voyelles
    moyennes = df.groupby('voyelles')[matiere].mean()
    counts = df.groupby('voyelles')[matiere].count()
    
    return moyennes, counts

# %% Visualisation pour toutes les matières
plt.figure(figsize=(20, 15))

for idx, matiere in enumerate(matieres):
    moyennes, counts = analyser_matiere(vue_df, matiere)
    
    # Créer un DataFrame pour l'analyse
    analyse = pd.DataFrame({
        'moyenne': moyennes,
        'count': counts
    }).reset_index()
    
    # Filtrer pour n'avoir que les combinaisons avec au moins 3 étudiants
    analyse = analyse[analyse['count'] >= 3]
    
    # Calculer la corrélation
    correlation = np.corrcoef(analyse['count'], analyse['moyenne'])[0,1]
    
    # Créer le subplot
    plt.subplot(3, 2, idx+1)
    
    # Scatter plot
    plt.scatter(analyse['count'], analyse['moyenne'])
    
    # Ligne de régression
    z = np.polyfit(analyse['count'], analyse['moyenne'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(analyse['count'].min(), analyse['count'].max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.xlabel('Nombre d\'étudiants')
    plt.ylabel('Moyenne')
    plt.title(f'{matiere}\nCorrélation: {correlation:.3f}')

plt.tight_layout()
plt.show()

# %% Tableau récapitulatif
resultats = []
for matiere in matieres:
    moyennes, counts = analyser_matiere(vue_df, matiere)
    analyse = pd.DataFrame({
        'moyenne': moyennes,
        'count': counts
    }).reset_index()
    analyse = analyse[analyse['count'] >= 3]
    
    correlation = np.corrcoef(analyse['count'], analyse['moyenne'])[0,1]
    
    # Trouver les meilleures et pires combinaisons de voyelles
    meilleure = analyse.nlargest(1, 'moyenne')
    pire = analyse.nsmallest(1, 'moyenne')
    
    resultats.append({
        'matiere': matiere,
        'correlation': correlation,
        'meilleure_voyelles': meilleure['voyelles'].iloc[0],
        'meilleure_moyenne': meilleure['moyenne'].iloc[0],
        'pire_voyelles': pire['voyelles'].iloc[0],
        'pire_moyenne': pire['moyenne'].iloc[0]
    })

# Afficher le tableau récapitulatif
resultats_df = pd.DataFrame(resultats)
print("\nTableau récapitulatif:")
print(resultats_df.to_string(index=False))

# %% Sauvegarder les résultats
resultats_df.to_csv('resultats_voyelles.csv', index=False)