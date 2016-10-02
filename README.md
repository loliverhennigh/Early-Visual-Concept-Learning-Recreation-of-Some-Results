# [Early-Visual-Concept-Learning-Recreation-of-Some-Results](https://arxiv.org/pdf/1606.05579v3.pdf)
Here is an implementation of a few results seen in Early Visual Concept Learning with Unsupervised Deep Learning. This paper looks at training variational autoencoders to produce "disentangled" representations in the latent space. Their main experiment involves learning black and white images of different shapes such as hearts, circles and boxes in different x, y positions and sizes. They find that by tuning a constant beta that controls the KL loss they can learn good disentangled representations. This results in latent variables corresponding directly to x, y positions and sizes. In this project, I have attempted to recreate these results on a dataset of black and white images of balls such as these ![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/real_balls_one_balls.jpg) (one ball dataset) ![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/real_balls_two_balls.jpg) (two ball dataset).

## Model

There are three models tested on both the one ball and the two ball dataset. The models used are fully connected, convolutional and all convolutional. The fully connected model is a replica of the model used in the paper. The only differences are the optimizer and the use of ELU instead of ReLU. The convolutional network was custom made to test how it does against the fully connected. The all convolutional model has no fully connected layer and keeps the latent encoding as a image. This seemed like a neat idea but turned out to not perform very well. For exact details on the models look at `architecture.py`.

## Training

To execute all experiments run `./run`. This will run each of the 3 models on the 2 datasets with 3 different values of beta. This totals 18 experiments. All models will get saved in `checkpoints`. Running this on a GTX 1080 took approximately 1 day for all experiments to finish. To create the figures presented, run `create_cool_images.py`, `create_error_graph.py`, and `create_stddev_graphs.py`. All plots are saved in `figures`. There are different parameters that can be messed with and are all flags in `main.py`.

## Results

There are three different kinds of figures produced.

### Disentangled Representation

In Figure 2 of the paper they show the effect of changing individual latent variables on the produced images. Doing this reveals that some of the latent variables correspond with things like position and size. Here are recreations of similar results for different models on the datasets. Using their same model I was able to get similar but slightly worse results.

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/heat_map_model_fully_connected_num_balls_1_beta_1.0.png)

In this figure we take on image of the ball and produce a latent encoding. Then we order which variables have the lowest to highest standard deviation and change variables individually between -3 and 3. This shows that only 3 variables have effect on the balls position. It seems as though these variables roughly correspond to some axis of ball position. For example the second variable roughly control the y position. The two ball dataset produces more entangled states

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/heat_map_model_fully_connected_num_balls_2_beta_1.0.png)

It looks as though the variables learn rotations of the two balls next to each other. The second variable twists the two balls in a clockwise fashion.

Over all, these are pretty good results. The other models produced more entangled encoding and all figures are included in `figures`. The all convolutional model was particularly bad and failed to learn disentangled representations.

### Standard Deviation

Here are the plotted average standard deviation of the latent encoding ordered from lowest to highest. This indicates how much the model is using each variable.

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/one_ball_stddev_fully_connected.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/one_ball_stddev_conv.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/one_ball_stddev_all_conv.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/two_ball_stddev_fully_connected.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/two_ball_stddev_conv.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/two_ball_stddev_all_conv.png)

We see from these plots that the fully connected model from the paper produced the best results only relying on a few variables heavily for the encoding. We also notice that increasing beta causes the latent variables to be used less. This makes sense because it increases the KL loss causing a stronger push for variables to learn 0 mean and 1 standard deviation. The same results are seen in the paper.

### Reconstruction Error

Any discussion of an autoencoder would be incomplete without a look at how well it reconstructs the input. Here are two figures that show how well does for each beta value and each dataset. 

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/one_ball_reconstruction_loss.png)

![alt text](https://github.com/loliverhennigh/Early-Visual-Concept-Learning-Recreation-of-Some-Results/blob/master/figures/two_ball_reconstruction_loss.png)

Again the fully connected model wins out on all tests. The trend observed in the paper about high beta values yielding high reconstruction error is confirmed in these results.

## Conclusion

This was a fun little project. I had no idea variational autoencoders where making these kinds of latent space encodings. I was really hoping that the all convolutional model would win over the fully connected in terms of reconstruction error but no such luck. It makes sense that it would have a significantly harder time producing disentangled states though. Keeping the latent encoding convolutional would make each variable effected differently by the image and probably make disentangling impossible.



