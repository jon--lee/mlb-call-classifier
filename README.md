# mlb-call-classifier
Umpire's aren't perfect. It's impossible to call pitches correctly
100 percent of the time. A lot of pitches end up on the corners and edges
of the strike zone. But can we find patterns in umpires' calls?
Do some umpires tend to call low pitches strikes and high pitches balls?
This classifier uses a supervised machine learning algorithm with `neuralpy` and MLB GameDay Data to develop a graphical representation of how an umpire sees his or her *unique strikezone.*

Here are some examples. where the blue, `0`, represents the strikezone and red, `1`, represents the area
outside the strikezone.

Average heights of bottom and tops of strike zone: http://www.baseballprospectus.com/article.php?articleid=14098

#### CB Bucknor
Training set: 437 examples: 68% used for training, 32% for test validation.  
Hyperparameters: 2-7-1 network with 200 epochs, 0.05 learning rate.  
Accuracy: 94% on test examples.

![](results/bucknor-94.png)

#### Bill Miller
Training set: 516 examples: 68% used for training, 32% for test validation.  
Hyperparameters: 2-7-1 network with 200 epochs, 0.05 learning rate.
Accuracy: 94% on test examples.

![](results/miller-94.png)