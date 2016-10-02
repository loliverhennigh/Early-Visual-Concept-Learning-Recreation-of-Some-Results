# [Early-Visual-Concept-Learning-Recreation-of-Some-Results](https://arxiv.org/pdf/1606.05579v3.pdf)
Here is an implementation of a few results seen in Early Visual Concept Learning with Unsupervised Deep Learning. This paper looks at training variational autoencoders to produce "disentangled" representations in the latent space. Their main experiment involves learning black and white images of different shapes such as hearts, circles and boxes in differents x, y posistions and sizes. They find that by tuning a constant beta that controls the KL loss they can learn good disentangled representations. This results in latent variables corrisponding to directly to x, y posistions and sizes. In this project I have attempted to recreate these results on a dataset of black and white images of balls such as these ![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/real_balls_one_balls.jpg) (one ball dataset) (two ball dataset).

## Model

There are three models tested on both the one ball and the two ball dataset. The models used are fully connected, convolutional and all convolutional. The fully connected model is a replica of the model used in the paper. The only differences are the optimiser and the use of ELU instead of ReLU. The convolutional network was custom made to test how it does agianst the fully connnected. The all convolutional model has no fully connected layer and keeps the latent encoding as a image. This seemed like a neat idea but turned out to not perform very well. For exact details on training look at `architecture.py`.

## Training

To execute all experiments run `./run`. This will run each of the 3 models on the 2 datasets with 3 different values of beta. This totals 18 experiments. All models will get saved in `checkpoints`. Running this on a GTX 1080 took approximently 1 day for all experiments to finish. To create the figures presented run `create_cool_images.py`, `create_error_graph.py`, and `create_stddev_graphs.py`. All plots are saved in `figures`. There are different parameters that can be messed with and are all flags in `main.py`.

## Results





