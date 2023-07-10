#streamlit run dashboard.py

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import re
import math
import base64
from zipfile import ZipFile
from lightgbm import LGBMClassifier
from streamlit_echarts import st_echarts
import requests

#Configuration application
# Titre de la page
st.markdown("""# Accord prêt bancaire: Analyse détaillée""")

#Affiche une ligne horizontale pour séparer le titre du contenu suivan
st.markdown("***") 

# Affiche un paragraphe de texte décrivant le but de l'application
st.write("""Cette application prédit la probabilité qu'un client de la banque "Prêt à dépenser" ne rembourse pas son prêt.
""")


# Mise en place du model, des données et du seuil
lgbm = pickle.load(open('best_final_prediction.pickle', 'rb'))

z = ZipFile("df_test_imputed.zip")
data_clean = pd.read_csv(z.open('df_test_imputed.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
all_id_client = data_clean.index

z = ZipFile("real_data_clean_test.zip")
data_origin = pd.read_csv(z.open('real_data_clean_test.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

z = ZipFile("test_imputed_without_standardisation.zip")
data_clean_without_standard = pd.read_csv(z.open('test_imputed_without_standardisation.csv'), index_col='SK_ID_CURR', encoding ='utf-8')


seuil= 0.6
# ajouter un sous titre pour afficher le seuil 
original_title = f'<p style="font-family:Courier; color:Blue; font-size: 18px;">La probabilité maximale de défaut de remboursement autorisée par la banque est de : {seuil}</p>'
st.markdown(original_title, unsafe_allow_html=True)

####################################################################################
#sidebar : Analyse descriptive pour chaque client
# créer une barre latérale "Analyse générale"
ori_title = '<p style="font-family:Courier; color:Red; font-size: 25px;text-align: center;">Analyse générale</p>'
st.sidebar.markdown(ori_title, unsafe_allow_html=True)

# choix du client en fonction de la liste disponible

def classify_client(model, ID, df, seuil):
    ID = int(ID)
    X = df[df.index == ID]
    #X = X.drop(['TARGET'], axis=1) #if df_train_imputes.csv
    probability_default_payment = model.predict_proba(X)[:, 1]
    if probability_default_payment >= seuil:
        prediction = "Prêt NON Accordé"
    else:
        prediction = "Prêt Accordé"
    return probability_default_payment, prediction

# choix du client en fonction de la liste disponible 

choice = st.sidebar.radio("Choix de l'identifiant client", ('Saisir un identifiant', 'Sélectionner dans la liste'))
id_client = ""

if choice == 'Saisir un identifiant':
    id_client = st.sidebar.text_input("Veuillez entrer l'identifiant d'un client")
elif choice == 'Sélectionner dans la liste':
    id_client = st.sidebar.selectbox("Liste de clients disponibles", all_id_client)


if id_client != "":
    # Vérifier si l'ID client n'est pas une chaîne vide
    id_client = int(id_client)  # Convertir l'ID client en entier
    if id_client not in all_id_client:
        # Vérifier si l'ID client n'est pas présent dans la liste des ID client
        st.write("Ce client n'est pas répertorié")  # Afficher un message indiquant que le client n'est pas répertorié
        st.text("")  # Ajouter une ligne vide pour l'espace

    else: # Si l'ID client est présent dans la liste des ID client
        # Récupérer les informations d'identité du client à partir du dataframe data_origin
        identite_client = data_origin[data_origin.index == int(id_client)]
        # Récupérer les informations de prédiction du client à partir du dataframe data_clean
        predict_client = data_clean[data_clean.index == int(id_client)]
        # Récupérer les informations du client sans normalisation à partir du dataframe data_clean_without_standard
        client_without_standard = data_clean_without_standard[data_clean_without_standard.index == int(id_client)]

        st.text("")
        st.text("")

        #st.markdown(original_title, unsafe_allow_html=True)


        st.write("**Genre : **", identite_client["CODE_GENDER"].values[0])
        st.write("**Age : ** {:.0f}".format(identite_client["AGE"].values[0]), 'ans')

        #Age distribution plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_origin["AGE"], color="orchid", bins=20)
        ax.axvline(int(identite_client["AGE"]), color="red", linestyle='dashed')
        ax.set(title='Age des clients', xlabel='Age (ans)', ylabel='')
        st.pyplot(fig)

        # Situation du client
        st.write("**Status Familial : **", identite_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfant : **{:.0f}".format(identite_client["CNT_CHILDREN"].values[0]))
        st.write("**Possession d'une voiture : **", identite_client["FLAG_OWN_CAR"].values[0])
        st.write("**Possession de votre propre logement : **", identite_client["FLAG_OWN_REALTY"].values[0])
        st.write("**Type de logement habité : **", identite_client["NAME_HOUSING_TYPE"].values[0])
        st.write("**Revenu total (USD) : **{:.2f}".format(identite_client["AMT_INCOME_TOTAL"].values[0]))

        st.text("")
        st.text("")

        df_income = pd.DataFrame(data_origin["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        # revenu client distribution plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_income["AMT_INCOME_TOTAL"], color="orchid", bins=20)
        ax.axvline(int(identite_client["AMT_INCOME_TOTAL"].values[0]), color="red", linestyle='dashed')
        ax.set(title='Revenu des clients', xlabel='Revenu (USD)', ylabel='')
        st.pyplot(fig)

        st.write("**Type de revenus : ** ", identite_client["NAME_INCOME_TYPE"].values[0])

        st.write("**Nombre de demandes de prêt : ** {:.0f}" .format(identite_client["PREVIOUS_APPLICATION_COUNT"].values[0]))
        st.write("**Type de prêt le plus demandé : **", identite_client["MOST_CREDIT_TYPE"].values[0])
        st.write("**Nombre de prêts accordés précédents : **{:.0f}".format(identite_client["PREVIOUS_LOANS_COUNT"].values[0]))
        st.write("**Montant total à payer pour les crédits :** {:.2f}".format(identite_client["AMT_CREDIT"].values[0]))
        st.write("**Pourcentage remboursement crédit sur les revenus total : **{:.2f}".format(identite_client["ANNUITY_CREDIT_PERCENT_INCOME"].values[0]*100), "%")
        st.write("**Montant du crédit remboursé par an : **{:.2f}".format(identite_client["AMT_ANNUITY"].values[0]))
        st.write("**Durée remboursement crédit en année : **{:.2f}".format(identite_client["CREDIT_REFUND_TIME"].values[0]))

        st.text("")
        st.text("")

        if id_client != "":
            
            probability_default_payment, prediction = classify_client(lgbm, id_client, predict_client, seuil)
            original_title = '<p style="font-size: 20px;text-align: center;"> <u>Probabilité d\'être en défaut de paiement : </u> </p>'
            st.markdown(original_title, unsafe_allow_html=True)

        #probability_default_payment, prediction = classify_client(lgbm, id_client, data_clean, seuil)
        #original_title = '<p style="font-size: 20px;text-align: center;"> <u>Probabilité d\'être en défaut de paiement : </u> </p>'
        #st.markdown(original_title, unsafe_allow_html=True)

# mise en place d'un graphique en jauge pour la probabilité d'étre en défaut de paiement 
        options = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},  # Définition du format du tooltip
            "series": [
                {
                    "name": "défaut de paiement",  # Nom de la série
                    "type": "gauge",  # Type de graphique
                    "axisLine": {
                        "lineStyle": {
                            "width": 10,  # Largeur de la ligne de l'axe
                        },
                    },
                    "progress": {"show": "true", "width": 10},  # Définition de la barre de progression
                    "detail": {"valueAnimation": "true", "formatter": "{value}"},  # Configuration du détail
                    "data": [{"value": (probability_default_payment[0]*100).round(2), "name": "Score"}],  # Données à afficher
                }
            ],
        }
        st_echarts(options=options, width="100%", key=0) # afficher le graphique 

        # conclusion 
        original_title = '<p style="font-size: 20px;text-align: center;"> <u>Conclusion : </u> </p>'
        st.markdown(original_title, unsafe_allow_html=True)

        # Afficher un texte pour le prêt accordé ou refusé en fonction du seuil
        if prediction == "Prêt Accordé":
            original_title = '<p style="font-family:Courier; color:GREEN; font-size:70px; text-align: center;">{}</p>'.format(prediction)
            st.markdown(original_title, unsafe_allow_html=True)
        else :
            original_title = '<p style="font-family:Courier; color:red; font-size:70px; text-align: center;">{}</p>'.format(prediction)
            st.markdown(original_title, unsafe_allow_html=True)
        
        st.text("")
        st.text("")

        #Feature importance / description 
        original_title = '<p style="font-size: 20px;text-align: center;"> <u>Quelles sont les informations les plus importantes dans la prédiction ?</u> </p>'
        st.markdown(original_title, unsafe_allow_html=True)
        feature_imp = pd.DataFrame(sorted(zip(lgbm.booster_.feature_importance(importance_type='gain'), data_clean.columns)), columns=['Value','Feature'])

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
        ax.set(title='Importance des informations', xlabel='', ylabel='')
        st.pyplot(fig)
