
Self Driving Car -(Deep Q Learning)

	Part 1 !

	In this part we are going to get started into the world of AI and buid out own setf driving car. THis is going to be a a modelled version of a car(so it wont be driving in the real cities) but still, it will learn how to drive itself and the key work here is learn because the car will not be given any rles on how to operate in the environment before hand - it will have to live everything out on its own. And to achieve that, we will be using deep-Q-learning. 

Deep q-Learning  :

	is the result of combining Q-Learning with an ANN. The states of the environment are encoded by a vector which is assesed as input into the neural network. Then the neural network will try to predict which action should be played, by returning as outputs a Q-Value for each of the possible actions. Eventually, the best action to play is chosen by either taking the one that has the highest Q-Value, or by overlaying a softmax function. 

We will implement a Deep-Q learning model to build an AI for a self driving car. 

Overall what the functions will do.

Deep Q-Learning intuition(learning)

	Update the network, pass them through the soft-max function.
	Take the loss and pass is again throug the Input layer SO that you update the network again.

Deep Q-Larning intuition(Acting)

	Pass the Q-Values to a softmax function. 
	Pass the Q-Values to the action-selection policies. 

Experience Replay

	Learning + Acting
	Vector of values , going through a neural network and output will be a couple of updated values. 
	A bunch of Q values. These values are passes throug the neural network, then the error values are calculated and then back propagated . The initial Q Values are again updated and then again fed to the neural network. 

	Various experiences do not get played into the network at once. They are saved in the memory. It randomly selects a uniformly distributed sample from that batch of experiences that it has and it learns from it. 


Action Selection policies. 

	how Deep learning agents are able to do exploration with exploitation.
	We want to comeup with a action selecction policy so that we do not get stuck in a loop. We have 3 AS
		e-greedy   
	We do not want to be stuck in local Max, thats why we need to keep on exploring. Even if we are getting good results from the local-max


How to follow the code in this tutorial. 
	
	Simple check all the commits from the first one. The following steps have been mentioned to make it easy to follow

	1.> Import all the libraries.
	 