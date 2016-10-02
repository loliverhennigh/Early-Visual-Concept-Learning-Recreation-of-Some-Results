# [Early-Visual-Concept-Learning-Recreation-of-Some-Results](https://arxiv.org/pdf/1606.05579v3.pdf)
Here is an implementation of a few results seen in Early Visual Concept Learning with Unsupervised Deep Learning. This paper looks at training variational autoencoders to produce "disentangled" representations in the latent space. Their main experiment involves learning black and white images of different shapes such as hearts, circles and boxes in differents x, y posistions and sizes. They find that by tuning a constant beta that controls the KL loss they can learn good disentangled representations. This results in latent variables corrisponding to directly to x, y posistions and sizes. In this project I have attempted to recreate these results on a dataset of black and white images of balls such as these (one ball dataset) (two ball dataset).

## Model

There were three models tested on both the one ball and the two ball dataset. A fullyconnected network that has the same w



