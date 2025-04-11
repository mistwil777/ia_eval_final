# RAG Chatbot Project

Ce projet implémente un chatbot RAG (Retrieval-Augmented Generation) utilisant Python, Ollama/LM Studio, LangChain, ChromaDB et sentence-transformers.

## Configuration de l'environnement

1. Installez les dépendances requises :
   ```
   pip install -r requirements.txt
   ```

2. Si vous rencontrez des problèmes avec sentence-transformers, exécutez le script de correction :
   ```
   python fix_huggingface.py
   ```

3. Vérifiez que l'environnement est correctement configuré :
   ```
   python setup_env.py
   ```

## Préparation des documents

1. Créez un répertoire `corpus` à la racine du projet
2. Ajoutez vos documents texte (.txt) dans ce répertoire

## Utilisation du chatbot

Lancez l'interface en ligne de commande :
```
python chatbot_cli.py --model llama2 --corpus ./corpus
```

Options disponibles :
- `--model` : modèle LLM à utiliser (défaut: llama2)
- `--corpus` : chemin vers le répertoire des documents (défaut: ./corpus)
- `--embedding_model` : modèle d'embedding à utiliser (défaut: all-MiniLM-L6-v2)
- `--results` : nombre de documents à récupérer (défaut: 4)

## Évaluation du chatbot

Exécutez l'évaluation avec 5 requêtes complexes prédéfinies :
```
python evaluation.py
```

Les résultats seront enregistrés dans le dossier `evaluation_results`.

## Structure du projet

- `fix_huggingface.py` : Script pour corriger les problèmes d'importation dans sentence-transformers
- `setup_env.py` : Vérifie la configuration de l'environnement
- `document_processor.py` : Gère le traitement et l'indexation des documents
- `rag_chatbot.py` : Implémentation du chatbot RAG
- `chatbot_cli.py` : Interface en ligne de commande
- `evaluation.py` : Script d'évaluation

## Composants du système RAG

1. **Traitement des documents** : Chargement, découpage en chunks et indexation
2. **Génération d'embeddings** : Conversion des textes en vecteurs avec sentence-transformers
3. **Stockage vectoriel** : Utilisation de ChromaDB pour stocker et rechercher les embeddings
4. **Recherche sémantique** : Récupération des passages les plus pertinents
5. **Prompt engineering** : Intégration du contexte pertinent dans les requêtes au LLM
6. **Interface utilisateur** : CLI pour interagir avec le système

