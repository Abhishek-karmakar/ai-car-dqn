
[![Build Status](http://circleci-badges-max.herokuapp.com/img/Abhishek-karmakar/ai-car-dqn?token=)](https://circleci.com/gh/Abhishek-karmakar/ai-car-dqn)


## Self Driving Car -(Deep Q Learning)

	We are going to get started into the world of AI and build out own self driving car. This is going to be a a modelled version of a car(so it wont be driving in the real cities) but still, it will learn how to drive itself and the key work here is learn because the car will not be given any rles on how to operate in the environment before hand - it will have to live everything out on its own. And to achieve that, we will be using deep-Q-learning. 

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
	2.> Create the architecture of the Neural Network. Make a class 
		- Define init() which will initialize the Neural Network , defines the variable of the object. Defineing the Input Layer - 5 neurons, The hidden layer and the output layer. 
		- Foward Function :  A rectifier activation because its purely non linear function. 

		- How many neurons do we need in the hidden layer set to 30 after some experiments.

		Neural network has multiple layers between them input layer, hidden layer , output layer. These layers are supposed to connected to each other. This is achieved by using the self.fc1 = nn.Linear(input_size, 30) & self.fc1 = nn.Linear(30, nb_actions). In the __init__() function we need to initialize the neural network. and define it so that we know what is the size of the neural network.

		forward Propagation.  - It will activate the neurons but also and mostly will requtnr the Q value depending the Q Values. 
		nn.functional module contains all the functions . We'll use the Relu function. Its a rectifier function. Using wither Softmax. 
   

	3.> Implementing Experience Replay

		we'll make a Replay class which will contain a couple of methods which will do all the work related to Experience Replay. 
		Its based on Markov Decision Process. One time step is not enough for the model to understand long term corellation.
		We put last one 100 steps in the memory. The Whole Deep Q Lerning process much better. 
		3 function - 
			__init__ will initialize 100 transitions in the memory 
			
			push function , 
				Append a new transition in the memory.  
				So we do not cross more than 100 memory.
			 - It has some parameters and those params will have some state. 

			sample function to sample some transitions in the 100 samples. 
				Return some sample Random functions.
				We are taking some random samples from memory with fixed batch size. 
				we want our samples in the following state.
					state
					actions
					rewards

				So that we can wrap these vaiables in a PyTorch Variable. A variable which contains both a Tensor and a Gradient. 
				To be able to differentiate with respect to a Tensor, we need the strucutre of a tensor and a gradient.  
				Variable function will convert a Torch function to a variable and a gradient. 

				Eventually we get a list of batches which is well alligned and each batch is a PyTorch variable. 

				This function is used to sample the memory. To train a model better.

4.> Implement Whole Deep Q Learning Model. 
		
		- Making a class with Various Functions. 
			init()
			Select_actions() - #select correct action
			Learn()
			update()
			Score function
			Save
			Load

		- __init__() will take input_size and nb_action and the Gamma parameter which is the delay coeffecient .
			initialize the Gamma
			initialize the reward_window - It will be a sliding window of the mean of the last 100 reward which we will use to evaluate the evolution of the AI performance. Will be an empty list. 
			initialize the Neural Network - Model, an onject of the network class. 
			initialize the optimizer - Stocastic Optimizer.  (we can chose from many) , Creating the Object of the Adam Class. 
			initialize the last_state - A vector of 5 dimension. 3 signals of the 3 sensors. Orientation and - Orientation. It needs to be a torch tensor. We need to also make a fake dimension. 
			initialize last_action will be a simple number. initilize it to Zero. 
			initialize last_reward will be a float number. 

		select_action() will take two parameters self and input state.
			we are going to generate the distribution of probablilities wrt the 3 Q values which are input state1,2,or 3. One Probability for each Q value and they will sum up to One . We'll do all this with SoftMax which will also return One. We can explore using a Temperature parameter. You can use a Softmax instead of a orgmax because this will let the car explore than follow a set-path. 

		learn ()- We are going to train the Deep Neural Network inside your AI. Whole process of forward propagation and Backpropagation so we get our output. We'll get our target also. Then we will compare the Target to the error to get our error. Then we will backpropagate this error into the neural network and using stocatic Gradient Decent we will update the weights according to how much they contributed to the last error.    

		        def learn(self, batch_state, batch_next_state, batch_reward, batch_action): - marcov desision process is in the heart of DQN.

		        #max of the q values of the next state represented by index 0 according to all the actions which are represented by index 1

		        td_loss = temporal difference , chose the loss function Huber Loss. F.smooth_l1_loss

			At each iteration of the loop we need to reinitialize the optimizer. to backpropagate our loss to the Neural Network. 

		update() - This will update everything as soon as the AI reaches a new state. rewards, The connection that we are doing to use the . This function will take the last_reward and last_action to make the new action and this will be used in the update function of the map.py. THis is the connection between the Game (Environment) and the AI.

			The AI needs to start learning from its last 100 events. We are learning from some random samples of the memory. 

			reward window will take a mean of last 1000 

		score() function - We will compute the score of the sliding_window. We will compute the mean of all the values of the reward_window

		save() - function to save the model or save the brain of the car. We will save the model of the brain.

			just want to save the model and weights. 

		load() - function to load the  







