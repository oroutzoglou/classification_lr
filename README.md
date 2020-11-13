
Run an image classification with python logistic_regression_NN.py

In order for Gradient Descent to work you must choose the learning rate wisely.
In logistic_regression_NN.py we choose learning rate as 0.005 randomly. The learning rate  αα  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

for exploration of learning rate run:
python optimal_learning_rate.py
