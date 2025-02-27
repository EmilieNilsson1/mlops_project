# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [X] Check how robust your model is towards data drifting (M27)
* [X] Deploy to the cloud a drift detection API (M27)
* [X] Instrument your API with a couple of system metrics (M28)
* [X] Setup cloud monitoring of your instrumented application (M28)
* [X] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [X] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

72

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s204204, s204259, s224235, s224227, s224205

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the third-party framework 'TIMM' in our project, a PyTorch based package that offers pretrained models. We used the functionality of the package to load pretrained models and later finetune them on our dataset. The package helped us to quickly load and train models, which saved us time on training and allowed us to focus on other parts of the project. We have used the ResNet18 model with pretrained weights and changed the prediction head (the last linear layer) to a linear layer with random weights and an output dimension of 10, to fit with the 10 classes in the dataset. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used a `requirements.txt` file to manage our dependencies. The file was auto-generated using `pip freeze > requirements.txt`. To get a complete copy of our development environment, one would have to run the following commands:

conda create --name <env_name> --file requirements.txt

We have ensured that our project is compatible with both python 3.11 and 3.12 and windows-latest, macos-latest and ubuntu-latest. 

If the new member where to change something in the code, it would also be required to install `requirements_dev.txt`and `requirements_tests.txt`

Alternatively, we also provided a Dockerfile to containerize the application, ensuring an isolated and consistent environment. By running `docker build` and `docker run` commands, a new team member can quickly set up the exact same environment without manually managing dependencies, making it especially useful for deployment or cross-platform development.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template we have filled out the `configs`, `.github`, `src`, `data`, `reports`, `tests`, and  `dockerfiles` folders. We have removed the `notebooks` and `docs` folders because we did not use any notebooks and did not write any documentation in our project. We have added an `outputs` folder that contains the logs for our experiments saving every checkpoint, we used the `outputs` folder instead of the `models` folder. We also added a `templates` folder for the html document used for the frontend to the API. Overall the structure of our repository is similar to the cookiecutter template with a few added files such as `data_drift.py` to run certain parts more efficiently.


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used ruff for linting and black for formatting to keep everything similar. We agreed that we should comment on the code to make it easier to read and understand.

Linting and formatting ensures that the code looks the same throughout the entire project and using packages like ruff and black makes it easier to do this, because it therefore it not nessecary to think about it as much during the codingproces. Typing og documantation is important in larger projects because they make the code easier to understand and interpret and explain the flow of the code and reasoning behind it. The concepts are important in larger prrojects because it makes it easier to read and understand the code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have implemented 13 pytest tests and 1 load test. These tests focus on the most critical parts of our application, including the inference API, data handling, and model behavior. We validate the functionality of the data handler, check the dimensions of tensors in the model, and ensure the accuracy of predictions. The load test assesses the performance and reliability of the API under high traffic, ensuring scalability and robustness.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total coverage is 92% with the missing coverage being lines that are basically non-testable. Even if we had a 100% coverage we would not trust the code to be error free because we test some specific things and it is almost impossible to make unittests that ensures that all possible errors are covered.

Fx. we haven't done any test on different file types as we would not expect this. If we wanted to fully secure the code we would have to create so many test that the pytest run would be inefficient. We have here decided to only do relevant test on the code that would ensure it could run effeciently in our given case. Instead we have in the code itself checks to ensure the user are not able to create any mistakes.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and PRs in our project. In our group, each feature was developed on a separate branch. When the feature was completed, a PR was created to merge the feature branch into the main branch. This allowed us to work on different features simultaneously without interfering with each other's work and ensuring that the main branch remained stable and production-ready at all times. Before merging the PR, we ensured it had passed teh automated checks set up in our GitHub workflow, including pytest for running unit tests. This ensured that any new changes did not break existing functionality and met the project's quality standards. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not use DVC because it did not make sense for our project. Version control for data is most beneficial in scenarios where datasets are frequently updated, modified, or iteratively refined. In such cases, tools like DVC provide clear tracking of changes, ensure reproducibility, and facilitate collaboration by linking data versions to corresponding code updates. However, in our case, the data was static and did not require frequent revisions. One situation where it could have been useful would be if we added the images uploaded by users via the API for the classifier to process, analyze, and further train the dataset for improved accuracy.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We run continuos integration on multiple operating systems, python- and torch version: ubuntu latest, MacOS latest and windows latest, python 3.11 and 3.12, torch versions 2.5.1, 2.41.1, 2.3.1. For each combination we run all of our unit tests. Hereby it runs the pytest on each version to verify that the scripts are able to be run in different scenarios for the user. 

We make use of caching in our workflows to improve efficiency. For example, dependency installation is cached to avoid redundant downloads and speed up the process. This is especially helpful when testing multiple Python and Torch versions. 

We also made a configuation file that sets up a CI procces which is triggered by changes in our datafile. This trigger executes the train scrips and makes sure the model is always up to data with the newest data.

An example of a triggered workflow can be seen here: https://github.com/EmilieNilsson1/mlops_project/actions/workflows/tests.yaml

We also use pre-commit hooks as part of our continuous integration workflow to maintain good code practise, here we use `ruff` for linting and end-of-file-fixer and trailing-whitespace from the `pre-commit` repository amongst others. The pre-commit configuration is defined in the `.pre-commit-config.yaml` file. 

These tools help os ensure well-tested code and keep `main` working. 


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used Hydra to configure our experiments. We created a single config file that contained all the hyperparameters for our experiments - batch size, learning rate, seed, and epochs. To run an experiment, we would modify the `configs/train.yaml` and use the following command:

`python src\image_classifier\train.py` or `python src\image_classifier\train.py --run local` to run the training locally instead of in the cloud.



### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of hydras config files to secure that no information is lost when running experiments and that our experiments are reproducible. Whenever an experiment is run, the config file is saved in the `outputs` folder. To reproduce an experiment, one would have to copy the config file into `train.yaml`and train the model again. 

Further more, as we use Weights and Biases to log our experiments, we save the config file in Weights and Biases under configs for each run using: 
`pl.loggers.WandbLogger(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), config=dict(cfg))`
As the `outputs` folder is kept local this allows the team to track all experiments as Weights and Biases can be used for team collaboration. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

As seen in the first image we have tracked the training loss, validaition loss and training accuracy and validation accuracy of the model. We track both training and validation to ensure we don't overfit on the data. The loss and accuracy are important metrics to track because they inform us about how well the model is performing. A decreasing training loss indicates that the model is improving its performance on the training data. Validation loss, on the other hand, measures the model's error on unseen validation data. It helps us understand how well the model generalizes to new data. A decreasing validation loss indicates good generalization, while an increasing validation loss may suggest overfitting. Training accuracy helps us understand how well the model is learning from the training data, while validation accuracy indicates how well the model generalizes to unseen data.

![WandB_loss](figures/WandB_loss.png)

In the second image, we have also included an image of how the hyperparameters are logged in Weights and Biases. Tracking the hyperparameters is important because it allows us to understand the impact of different configurations on the model's performance. By logging hyperparameters, we can easily compare different experiments and identify the best settings for our model.

![Wand_conf](figures/wand_config.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We have used docker both for local development but also especially for deployment in te cloud. We have used docker in Cloud Run to deploy apis and when running Cloud Engine for training the model.

We have seperate dockerfiles; one for training, one for inference api and one for data drift api. Each of these docker images download the requirements and copies the relevant files and run the appropriate entry point script tailored for its specific purpose.

To eg run training in docker, one would run 
`docker run train:latest` 

link to one docker file: https://github.com/EmilieNilsson1/mlops_project/blob/main/dockerfiles/train.dockerfile

when working in the cloud we have saved our docker images to the artifact registry, and from here it can be run like:
`docker run europe-west1-docker.pkg.dev/endless-galaxy-447815-e4/my-container/artifact-image:latest`

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We tried finding the bug and then used chatgpt or copilot or TA's to help debug the code. We occasionally used vscodes debugger, but didn't use it in its full potential.

When encountering errors in Google Cloud we used the logs to debug. 

We did try profiling some of our code, but didn't change anythings as it was not our main priority to optimize the code. Since we were using a pretrained model, most of the heavy lifting was already handled by PyTorch’s optimized back-end operations. Additionally, as we were using PyTorch Lightning, a lot of the boilerplate and optimization-related tasks were managed for us. If we were to optimize something it would be the data handling and dataloader. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We made use of the following GCP services in our project: Engine, Bucket, Artifact Registry, Build, Run. Engine is used for running virtual machines in the cloud. Bucket is used for storing data in the cloud. Artifact Registry is used for storing docker images in the cloud. Build is used for building docker images in the cloud. Run is used for deploying apis. 

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to test our training jobs, when the code ran without errors, we trained multiple models at the samme time using costum jobs in Vertex AI. We used an instances with the following hardware: n1-highmem-8, which has 8 vCPUs and 52 GB memory. This machine type is designed for memory-intensive workloads, however on further inspection it would probably have been wise to use a VM with GPU for faster model training. 
To ensure a consistent and tailored environment, we started the instances using a custom container. The container image was pulled from our private artifact registry at: europe-west1-docker.pkg.dev/endless-galaxy-447815-e4/my-container/artifact-image:latest.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Bucket](figures/image.png)


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Artifact registry](figures/image-1.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Cloud build](figures/image-2.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model in the cloud using Vertex AI. We did this by creating a custom container with our code and pushing its docker image to the Artifact Registry. We then created a custom job in Vertex AI that used the custom container to train the model. The reason we choose Vertex AI was because it allowed us to easily train our model in the cloud without having to worry about setting up the infrastructure ourselves. We had a few problems with the setup at first but ended up fixing it and created working models which was saved in the bucket together with the training data.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We used FastAPI to write the API. We made an API that can take an image as input and then use our trained model to predict which animal is shown in the image. We also created frontend so it is possible to upload an image from the computer and then get the prediction. Together with this the api frontend made it possible for the user to view the image they would use and prior ones. Here we also saved the predcitions to the cloud for later use. We didn't focus too much on the design of the frontend but more on it's usability to the user.
### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We deployed our API locally first. To run the API locally the user would call: 
`uvicorn src.image_classifier.api:app --reload`

Then the user can access the API on http://127.0.0.1:8000 and upload an image.


Here we did most of the testing finding faults and bugs in the code that could be of detrement to the user. With the knowledge that the api could run locally with all the decired features and some checks to make the code more stable we decided to run it on the cloud to make it easier to access.

Afterwards we deployed it to the cloud using Cloud Run by building a docker image, the service can be used following this link https://app-918145500414.europe-west1.run.app

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We did both unit testing and load testing of the API. For unit testing, we used pytest to test various functionalities of the API. Specifically, we tested the root endpoint to ensure it is accessible and returns the correct status code and content. We also tested the predict endpoint by uploading an image and verifying that the response contains the expected prediction and uploaded image information. For load testing, we used Locust to test the predict endpoint of the API by simulating multiple users uploading images for prediction. The results of the load test showed that the API is performing well with no failures and reasonable response times, indicating that it can handle a significant number of concurrent requests without issues.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.* 
>
> Answer:
We have 3 alert policies to watch if the the bucket is filled with requests and if too many files are located in the bucket. This is done to check if anyone abuses the api or likewise. We also check the burn rate of the project to see if the uptime is high enough. The monitoring here is setup so we can watch these parts of the project more closely and would hereby be able to focus on the less optimized parts.

If we had more time with the project we would have created more policies to protect the application from unforseen issues. This would also be usefull to optimize the application when running it in the cloud.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We have used around 28 dollars total. The training was the thing that was most expensive seeing as we did it wrong the first time and didn't save it correctly. This meant that we had to do it five times using a lot of credits. This was done in VertexAI accounting for 10 dollars. Another 10 dollars was used on Cloud storage, and the remaining was used on Artifact, Run and Engine. Working in the cloud was frustrating at times, but we do see how it can be very useful, especially the option for data storage in Cloud storage as we don't know of an alternative to this. 

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

As previously mentioned in question 23 we also made a frontend for the API, to make it easier to use. The frontend makes it possible to upload an image and get a prediction by implementing an HTML file.

We also implemented data drifting. We calculated brightness, contrast, mean and std for each channel on our training data and used this as baseline. When new images are uploaded to the deployed inference API the predicted images are saved to the bucket. By calling the data drift API, we can compare the baseline metrics to the prediction images metrics. We also managed to deploy this in Cloud Run: https://drift-918145500414.europe-west1.run.app/report

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![Pipeline](figures/image-3.png)

The starting point here in our pipeline is the local device of the devs which is where the code is created and training data is kept originaly. From this point we have pre-commit check on new code followed by commit and push before anything is uploaded to Github. When pushed to GitHub an Workflow script auto triggers the pytest which test if the scripts are able to be run on more systems and version. The devs can manually upload the training data to Google Cloud Storage in a bucket where it is easily accessed. 
When code is pushed to the github and auto trigger then sent the repository to Google Cloud where a container is created from which a docker image is made and able to be pulled by the user. In the Google Cloud we are then able to use the image and the training data to train models in Ai Vertex from which the model is saved to artifact registry and then converted to a useable model in the same bucket. This can then be run by a Google Cloud run as an api which creates a frontend able to be used by users. This api then saves the tested images in the bucket with it's prediction.


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

We have spent the most time trying to set things up on cloud and waiting for our model to train seeing as we also did the training multiple times. We also struggled with other functionalities of the Cloud, and especially how to combine these different services. It was a struggle to find the data in the bucket when training in cloud, and finding the model checkpoint when deploying the inference api. We found it difficult to debug in cloud aand also frustrating to have to wait a long time for an error. To overcome the challenges we have asked the TA's or chatgpt for help.

Another struggle was delegating the different tasks in the project, as many tasks builds on one another, eg it is hard to work on training in the cloud if the bucket is not set up, or the final training script is not done. To overcome this we worked a lot with mock data, ie we uploaded a few images to the bucket or an old checkpoint such that we could continue working on other tasks while waiting for the dependencies to be fully completed. This allowed us to make progress on separate components of the project without being blocked by unfinished elements, ensuring that we could test parts of the workflow and stay on schedule.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s204204 was in charge of setting the initial cookie cutter project, writting model and train scripts, setting up the continous integration, checking for datadrifting and deploymen

Student s204259 was in charge of the unittesting, monitoring and the dataprocessing.

Student s224235 was in charge of setting up the google cloud bucket, training our models on the cloud and deploying them and building docker images. 

Student s224227 was in charge of wrtting the code for the dataprocessing, uploading the data to the could, writing the API and creating the API-testing. 

Student s224205 was in charge of continous integration, the workflow triggers and building docker images.

All members contributted to code and debug the different areas of the project.

We have also used ChatGPT and Copilot to help debug and write some of the code.
