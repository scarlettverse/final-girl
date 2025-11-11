# 🩸 Final Girl
Predicting TV show survival with machine learning, not every series gets a sequel.

<br>

## 🎬 Problem Statement
Every year, dozens of TV shows enter the lineup. Some get renewed. Most get canceled. The question is: can we see it coming?

**Final Girl** is a machine learning experiment that asks whether we can predict a show’s survival based on its metadata including genre, ratings, network, and other features that might signal plot armor. Think of it like teaching a computer to spot the scream queen before the opening scene.

Inspired by the horror trope of “final girl”, the last one standing after everyone else gets slashed, we’ll treat renewal as survival. The model becomes a survivor’s Randy from Scream—knows the rules, spots the patterns, and predicts who’s next.

We’ll train a simple classification model, compare a few approaches, and see how well it can predict a show’s fate. But the goal isn’t just prediction. It’s understanding what makes a show resilient in a world of ruthless cancellations.

<br>

## 🧠 How Machine Learning Helps
It finds the survival patterns, ranks the tropes, and predicts who gets slashed!

- **Finds Survival Patterns:** It scans metadata like genre, rating, and network to figure out who’s most likely to survive the season.  
- **Ranks the Tropes:** It ranks features like whether horror shows last longer than comedies or if certain networks protect their scream queens.
- **Predicts who Gets Slashed:** It predicts whether a show gets renewed or canceled, like a slasher deciding whose next.

<br>


## 🎥 Director’s Cut: Behind the Scream
From data slicing to model tuning, this is Final Girl's arsenal.

- pandas
- numpy
- scikit-learn
- jupyter
- matplotlib & seaborn

<br>


## 📊 Dataset

I reviewed a few options and chose the [TMDb TV Shows Dataset](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows).It includes renewal status, rich but filterable metadata, and aligned with my machine learning goals.

<br>


## 🧪 Scope of Work from the Professor

- Pick a problem that interests you and find a dataset   
- Describe the problem and how ML can help  
- Prepare the data and run EDA  
- Train several models, tune them, and pick the best
- Export your notebook to a script
- Package your model as a web service and deploy it with Docker
