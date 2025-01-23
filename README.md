# Project description
The overall goal of this project is to build a simple image classifier, which can classify images of animals into 10 different categories: {dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, spider}.

We will be working with a pretrained model and will use the PyTorch timm library.

We will be using the dataset from Kaggle (https://www.kaggle.com/datasets/alessiocorrado99/animals10), consisting of the 10 classes mentioned above. The dataset consists of 26k labeled images.

We initially plan to use a pretrained ResNet18.

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
│       └── cane
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
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
