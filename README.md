# BeatAML CTD^2 DREAM Challenge: Example 2

Example implementation of a solution to subchallenge 2 of the BeatAML CTD^2 DREAM challenge. This example uses all input data types to train a Cox Model with ElasticNet regularization [1] to predict per-specimen hazard. 

## To train a model

* Run Jupyter with `docker run -p 8888:8888 -v "$PWD:/home/jovyan" jupyter/scipy-notebook`
* Stdout will include a URL to open the notebook
* Go through the steps in index.ipynb
* The model will be stored in model/ in a bunch of files
* Read more about the model [below](#the_model)


## To Run Your Model on Training Data

This model can be run on the same data it was trained on, to test whether the Dockerfile works:

```bash
SYNAPSE_PROJECT_ID=<...>
docker build -t docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model .
docker run -v "$PWD/training/:/input/" -v "$PWD/output:/output/" docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model 
```


## Submitting to Synapse DockerHub

```bash
SYNAPSE_PROJECT_ID=<...>
docker login docker.synapse.org
docker build -t docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model .
docker push docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model
```

## The Model

See `index.ipynb` for more explanation of the feature selection.


[1] Powered by [scikit-survival](https://scikit-survival.readthedocs.io/en/latest/index.html#).

