# snaQue-Learning Deep Reinforcement Learning
## Snake bot learns to play using Deep Q Learnining with Reinforcement Learning

- Neural Network:
	- Input layer: “Game state” (danger x3, direction x4, food x4) = 11 nodes

	- Output layer: “Action to take” (turn left, right or keep forward) = 3 nodes

	- Hidden layer: “learning” 256 nodes

- Deep Q Learning:
	- Initial Value (init Q function value)
	- Choose Q value (using model.predict() )
	- Mesure reward: (+50 eat food; -50 collision; -5 do nothing)
	- Return Action output value
	- Update Q value (training the NN model) -> repeat
