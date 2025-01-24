# Project description
The overall goal of this project is to build a simple image classifier, which can classify images of animals into 10 different categories: {dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, spider}.

We will be working with a pretrained model and will use the PyTorch timm library.

We will be using the dataset from Kaggle (https://www.kaggle.com/datasets/alessiocorrado99/animals10), consisting of the 10 classes mentioned above. The dataset consists of 26k labeled images.

We initially plan to use a pretrained ResNet18.

## Installation
Clone the repository using

`git clone https://github.com/EmilieNilsson1/mlops_project.git`

initialize an environment and install the dependencies using 

`conda create --name <env_name> --file requirements.txt`

## Usage
Firstly download the data from kaggle (if you want to run locally) and run `src/image_classifier/data.py` to preprocess the data.

To configure a new experiment change the hyperparameters in `configs/train.yaml` and add you wandb information to a `.env`. 

To run a training in the cloud with this command:
`python src/image_classifier/train.py` or simply `train`.
If you do not have access to our cloud the training can be run locally using the command:
`python src/image_classifier/train.py --run local`

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   └── workflows/
│       └── data_change.yaml
│       └── model_registry_update.yaml
│       └── tests.yaml
├── configs/                  # Configuration files
│   └── __init__.py/
│   └── train.yaml.py/
├── data/                     # Data directory
│   ├── processed
│       └── images
│   └── raw
│       └── cane
│       └── cavallo
│       └── elefante
│       └── farfalle
│       └── gallina
│       └── gatto
│       └── mucca
│       └── pecora
│       └── ragno
│       └── scoiattolo
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── artifact.Dockerfile
│   └── datadrift.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── image_classifier/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── translate.py
└── templates/
│   ├── index.html
└── tests/                    # Tests
│   ├── __init__.py
│   ├── loadTest.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── cloudbuild.yaml
├── config.yaml
├── data_drift.py
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
├── requirements_tests.txt    # Test requirements
└── tasks.py                  # Project tasks
└── vertex_ai_train.yaml      
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
