# Exercise Descriptions

## Differential Privacy and Federated Learning

Work in a group of 4 to accomplish this task.

Prerequisites:
1. Reddit dataset
2. "small" LLM 
3. Code from previous days
   1. Data processing
   2. Fine-tuning

Assignment:
1. Fine-tune language models on Reddit dataset
   1. Measure memorization of data
   2. Add canaries and retrain
2. Implement a vanilla version of Federated Learning
   1. Split dataset by individual users
   2. Implement Federated Averaging
   3. Add differentially private aggregation w noise 0.01 and clipping bound 1

## Safety Attack

Play a game where one team injects a factual error into the model and other team detects it.
Factual error has to be about machine learning.

Feel free to experiment with injection methods (training data, training algorithm) 
and detection methods: prompting, anomaly detection, etc. 


## Multi-modal adversarial attack (extension):

1. Use LLaVa model
2. Implement PGD attack of your target choice
3. Measure downstream task performance.