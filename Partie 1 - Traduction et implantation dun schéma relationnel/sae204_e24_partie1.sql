DROP SCHEMA IF EXISTS "partie1" CASCADE;
CREATE SCHEMA "partie1";
SET SCHEMA 'partie1';

-- Table _individu
CREATE TABLE _individu (
    id_individu INT PRIMARY KEY,
    nom VARCHAR(50) NOT NULL,
    prenom VARCHAR(50) NOT NULL,
    date_naissance DATE NOT NULL,
    code_postal VARCHAR(50),
    ville VARCHAR(50),
    sexe CHAR(1),
    nationalite VARCHAR(50),
    ine VARCHAR(50)
);

-- Table _candidat
CREATE TABLE _candidat (
    no_candidat INT PRIMARY KEY,
    classement VARCHAR(50) DEFAULT NULL,
    boursier_lycee VARCHAR(50),
    profil_candidat VARCHAR(50),
    etablissement VARCHAR(50),
    dept_etablissement VARCHAR(50),
    ville_etablissement VARCHAR(50),
    niveau_etude VARCHAR(50),
    type_formation_prec VARCHAR(50),
    serie_prec VARCHAR(50),
    dominante_prec VARCHAR(50),
    specialite_prec VARCHAR(50),
    lv1 VARCHAR(50),
    lv2 VARCHAR(50),
    id_individu INT NOT NULL,
    CONSTRAINT fk_candidat_individu FOREIGN KEY (id_individu)
   	 REFERENCES _individu(id_individu)
);

-- Table _etudiant
CREATE TABLE _etudiant (
    code_nip VARCHAR(50) PRIMARY KEY,
    cat_socio_etu VARCHAR(50),
    cat_socio_parent VARCHAR(50),
    bourse_superieur BOOLEAN,
    mention_bac VARCHAR(50),
    serie_bac VARCHAR(50),
    dominante_bac VARCHAR(50),
    specialite_bac VARCHAR(50),
    mois_annee_obtention_bac CHAR(7),
    id_individu INT NOT NULL,
    CONSTRAINT fk_etudiant_individu FOREIGN KEY (id_individu)
   	 REFERENCES _individu(id_individu)
);

-- Table _semestre
CREATE TABLE _semestre (
    id_semestre INT PRIMARY KEY,
    num_semestre CHAR(5) NOT NULL,
    annee_univ CHAR(9) NOT NULL
);

-- Table _inscription
CREATE TABLE _inscription (
    code_nip VARCHAR(50),
    id_semestre INT,
    groupe_tp CHAR(2),
    amenagement_evaluation VARCHAR(50),
    PRIMARY KEY (code_nip, id_semestre),
    CONSTRAINT fk_inscription_etudiant FOREIGN KEY (code_nip)
   	 REFERENCES _etudiant(code_nip),
    CONSTRAINT fk_inscription_semestre FOREIGN KEY (id_semestre)
   	 REFERENCES _semestre(id_semestre)
);

-- Table _module
CREATE TABLE _module (
    id_module CHAR(5) PRIMARY KEY,
    libelle_module VARCHAR(50),
    ue CHAR(2)
);

-- Table _programme
CREATE TABLE _programme (
    id_module CHAR(5),
    id_semestre INT,
    coefficient NUMERIC(5, 3),
    PRIMARY KEY (id_module, id_semestre),
    CONSTRAINT fk_programme_module FOREIGN KEY (id_module)
   	 REFERENCES _module(id_module),
    CONSTRAINT fk_programme_semestre FOREIGN KEY (id_semestre)
   	 REFERENCES _semestre(id_semestre)
);

-- Table _resultat
CREATE TABLE _resultat (
    code_nip VARCHAR(50),
    id_module CHAR(5),
    id_semestre INT,
    moyenne NUMERIC(4, 2),
    PRIMARY KEY (code_nip, id_module, id_semestre),
    CONSTRAINT fk_resultat_etudiant FOREIGN KEY (code_nip)
   	 REFERENCES _etudiant(code_nip),
    CONSTRAINT fk_resultat_module FOREIGN KEY (id_module)
   	 REFERENCES _module(id_module),
    CONSTRAINT fk_resultat_semestre FOREIGN KEY (id_semestre)
   	 REFERENCES _semestre(id_semestre)
);



