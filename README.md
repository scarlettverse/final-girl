![Final Girl Banner](assets/FINAL%20GIRL.png)

# 🩸 Final Girl
Predicting tv show survival with machine learning, not every series gets a sequel.

<br>

## 🎬 Problem Statement
Every year, dozens of TV shows enter the lineup. Some get renewed. Most get canceled. The question is: can we see it coming?

**Final Girl** is a machine learning experiment that asks whether we can predict a show’s survival based on its metadata including genre, ratings, network, and other features that might signal plot armor. Think of it like teaching a computer to spot the scream queen before the opening scene.

Inspired by the horror trope of “final girl”, the last one standing after everyone else gets slashed, we’ll treat renewal as our Final Girl. The model becomes a survivor’s Randy from Scream—knows the rules, spots the patterns, and predicts who’s next.

We’ll train a simple classification model, compare a few approaches, and see how well it can predict a show’s fate. But the goal isn’t just prediction. It’s understanding what makes a show resilient in a world of ruthless cancellations.

<br>

## 🧠 How Machine Learning Helps
It finds the survival patterns, ranks the tropes, and predicts the Scream Queens!

- **Finds Survival Patterns:** Scans metadata like genre, rating, and network to figure out who’s most likely to survive season two.  
- **Ranks the Tropes:** Ranks features like whether horror shows last longer than comedies or if certain networks protect their Final Girls.
- **Predicts the Scream Queens:** Predicts whether a show gets renewed or canceled, like a slasher deciding whose next.

<br>

## 📊 Dataset
Every slasher needs victims. Ours come from the [TMDb TV Shows Dataset](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows). It includes renewal status, rich but filterable metadata, and aligns with our machine learning goals.

### Key Features:
- **Status** → survived or canceled  
- **Genre** → horror, comedy, drama, etc.  
- **Network** → who protects their Scream Queens  
- **Votes & Popularity** → audience reception and plot armor  
- **Episodes** → longevity signals


This dataset gives us the cast list for our experiment: the Scream Queens who light up the screen, and the Final Girls who make it to season two

<br>

## 🎥 Director’s Cut: Behind the Scream
From organizing messy data to teaching the computer the rules of survival, this is our toolkit:

- **pandas** → data wrangling, the machete for messy tables  
- **numpy** → the math engine, the bones beneath the scream  
- **scikit-learn** → machine learning toolkit, the rules Randy whispers  
- **jupyter** → notebook stage, where the story unfolds  
- **matplotlib & seaborn** → visualization blades, turning numbers into blood‑red charts

<br>


## 🧼 Scripts Overview
Every slasher story gets rewritten. Our notebook became a set of scripts, each with its own role:

- **config.py** → the settings file, keeping the story consistent
- **training.py** → trains and saves the model, sharpening the blade  
- **predict.py** → loads the model and makes predictions, whispering who survives  
- **serve.py** → Flask API that exposes the model as a web service, the stage where the model performs
- **prepare_data.py** → sets up the dataset, cleaning and formatting the victims before the slasher arrives

<br>

## 📖 Usage: Fate Prediction
Final Girl is designed to lookup **show titles**. 
Enter a show title and the model will tell you if she’s the Final Girl or the next Scream Queen.

You can run predictions without Docker by calling the script directly:

`
Python scripts/predict.py "Lovecraft Country"
`
<br>

![The Solo Kill](assets/Examples/Lovecraft%20Country.png)

*Note: predictions are also saved to predictions.csv for auditing when ran locally.*

<br>

## 📈 Model Performance & Feature Insights
The model was trained, tested, and scored. Here’s how it performed:

- **Accuracy** → how often the model guessed survival correctly  
- **Precision** → how well it identified true survivors without false alarms  
- **Recall** → how well it caught all the survivors, even the hidden ones  
- **F1 Score** → the balance between precision and recall, the survivor’s final showdown  
- **AUC (ROC)** → how well the model separates survivors from victims, the slasher’s sharpest edge   

### 🔍 Feature Insights
The model also revealed which features mattered most:

- **Status** → survived or canceled
- **Genre** → horror vs. comedy survival rates  
- **Network** → some networks protect their final girls better than others  
- **Votes & Popularity** → audience reception as plot armor  
- **Episodes** → longevity signals that hint at resilience  

Together, these insights show which shows had the best chance of becoming Scream Queens and which ones had the plot armor to endure as Final Girls.

*Note: results differed slightly between notebook and script runs due to refactoring and pipeline changes.*

<br>

## 🩸 Setup Instructions: Opening Scene

Before the sequel plays out on Docker, here’s how to run the story locally:

Clone the Repo
```
Python
git clone https://github.com/scarlettverse/final-girl
cd final-girl
```
Install Requirements
`
Python
pip install -r requirements.txt
`

Run Scripts Locally
```
Python scripts/prepare_data.py
python scripts/training.py
python scripts/predict.py
```

<br>

## 🚀 Deployment: The Sequel
Every slasher gets a sequel, our model does too. We containerized the project with **Docker**, making it portable and reproducible across machines. The trained model is served through a **Flask API**, so anyone can send data and get predictions back.

Quick Start:
- Build the container:
`python
docker build -t final-girl .
`
- Run the container:
`python
docker run -p 5000:5000 final-girl
`  
- Test the API:
  ```
  bash
 
  # Title Mode
  curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Buffy the Vampire Slayer"}'
  ```

*Note: If you update the code, rebuild the Docker image using the steps above before running again*
<br>

## 🌐 Live API on Render: [Final Girl](https://final-girl.onrender.com)
The Final Girl API is deployed and live on Render. You can call it directly without running Docker locally.
- Link:
` https://final-girl.onrender.com/predict `

<br>

➡️ Scandal was judged a Scream Queen, with ~.08% chance of survival.
![The Solo Kill](assets/Examples/Title%20Model.png)

<br>


## 🧪 Scope of Work from the Professor

- Pick a problem that interests you and find a dataset   
- Describe the problem and how ML can help  
- Prepare the data and run EDA  
- Train several models, tune them, and pick the best
- Export your notebook to a script
- Package your model as a web service and deploy it with Docker
