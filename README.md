README - Metaheuristics for Feature Selection in Seismic Volcanic Activity Classification

Overview

This repository contains the source code and related materials for my thesis on using metaheuristic algorithms to select features for classifying seismic volcanic activity in Cotopaxi Volcano. The project leverages advanced machine learning techniques to enhance the accuracy of classification by optimizing feature selection through various metaheuristic approaches.

Thesis Objective

The primary goal of this research is to develop a method for classifying volcanic activity based on seismic signals, specifically focusing on Cotopaxi Volcano. This classification task involves selecting the most informative features from a vast dataset of 84 features. Instead of manually choosing features or using traditional statistical methods, I employed metaheuristic optimization techniques to identify the optimal subset of features that improve model performance.

The metaheuristcs used were Cuckoo Search, Genetic Algorithm, Simmulated Annealing, and ABACO. These methods aim to balance exploration and exploitation to ensure that the best possible features are chosen, thereby improving classification performance.

Repository Structure

FinalRuns: Contains final versions of all experiments, including results for each of the metaheuristic techniques applied.

Metaheuristicas: Holds the implementations of different metaheuristic algorithms used in feature selection. Algorithms such as Genetic Algorithm, Simulated Annealing, Cuckoo Search, and ABACO

Old: Includes older versions of scripts and preliminary tests that were conducted before finalizing the current approach.

Optimization_Runs: Contains scripts and configurations used during optimization runs. Each file in this folder represents an iteration or variation in the feature selection process.

Resources: Includes datasets the dataset utilized for this experimente

Methods

The metaheuristics in this research were used to select features from seismic data that could indicate volcanic events such as eruptions or increased activity. Each algorithm iteratively explored the feature space to find a subset that maximized classification accuracy when tested on machine learning models, including Gaussian Naive Bayes, Random Forest, and Neural Networks.



How to Use This Repository

Clone the Repository: Clone this repository to your local machine.

Navigate to the Folders: Start with the Metaheuristicas folder to explore the feature selection techniques.

Final Analysis: Use the FinalRuns folder to review final experiment results and observe the classification performance. Parameters can be easily modified in this documents

Requirements

Python:

Libraries: scikit-learn, numpy, pandas, matplotlib

Metaheuristics: Implementations use custom and standard Python libraries.

Contact



Acknowledgements



