# Project Overview

This repository contains the assignments for an Evolutionary Robotics course. The assignments explore various concepts in evolutionary computation and robotics, from simple Braitenberg vehicles to more complex tasks like neuro-evolution and optimization.

The project is structured into three main assignments:

*   **Assignment 1: Braitenberg Vehicles**
    *   This assignment implements a simple robot with two light sensors. The robot's behavior (aggressor or fear) is determined by how the sensor readings are mapped to wheel speeds.
    *   Key files: `Assignment-1/task-one.py`

*   **Assignment 2: Hill Climbing for Navigation**
    *   This assignment uses a hill-climbing algorithm to train a robot to navigate an arena with obstacles. The robot's controller is a simple neural network, and the fitness function is based on the area explored by the robot.
    *   Key files: `Assignment-2/hill_climber_proximity_sensor.py`

*   **Assignment 3: Optimization and Classification**
    *   This assignment is divided into two parts:
        1.  **Ackley Function Optimization:** An evolutionary algorithm is used to find the minimum of the 3D Ackley function.
        2.  **ANN Classifier:** An evolutionary algorithm is used to train a simple Artificial Neural Network (ANN) for a binary classification task.
    *   Key files: `Assignment-3/ackley-optimisation.py`, `Assignment-3/ex2/ex2_ann_classifier.py`

# Building and Running

The assignments are implemented in Python and rely on the following libraries:

*   `pygame` for the simulations and visualizations.
*   `matplotlib` for plotting the results.
*   `numpy` for numerical operations.
*   `pandas` for data manipulation in Assignment 3.

To run the assignments, you will need to have these libraries installed. You can install them using pip:

```bash
pip install pygame matplotlib numpy pandas
```

To run a specific assignment, execute the corresponding Python script. For example, to run the first assignment:

```bash
python Assignment-1/task-one.py
```

# Development Conventions

The code is well-commented and follows a clear structure. Each assignment is self-contained in its own directory. The code uses object-oriented principles where appropriate (e.g., for the robot and sensors). The use of `matplotlib` for plotting results is consistent across all assignments.
