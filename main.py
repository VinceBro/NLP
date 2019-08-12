from fastai import *
from fastai.text import *

class FrenchClassifier:
    def __init__(self):
        print("This program was made with research purposes and shows the possibilities of NLP with a language model trained from scratch, made in July 2019 by Momentum Technologies")
        print("Ce programme a été fait pour la recherche et démontre les possibilités de NLP avec un modèle de langue entrainé de rien du tout, fait en Juillet 2019 par Momentum Technologies")
        self.mdl_path = Path(input("Entrez le path relatif aux modèles : "))
        self.language_model= load_learner(path=self.mdl_path, file=input("Nom du modèle de prédiction : "))
        self.classifier= load_learner(path=self.mdl_path, file=input("Nom du modèle de classification: "))
        print("Models loaded")
        self.num_of_words = int(input("Définissez le nombre de mots pour la prédiction : "))
        self.temperature = float(input("Définissez la volatilité pour le modèle (entre 0 et 1) : "))


    def classify(self, text):
        return self.classifier.predict(text)
    
    def predict(self, text ):
        return self.language_model.predict(text, self.num_of_words, temperature=self.temperature)
    ## Load model


if __name__ == "__main__":
    fc = FrenchClassifier()
    while True:
        text = input("Entrez le texte pour la classification et la prédiction : ")
        print(f"\nSentiment : {fc.classify(text)}\n\n Prédiction : {fc.predict(text)}")