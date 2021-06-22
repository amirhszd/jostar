# Jostar
## _Feature Selection Module for Data Sciences in Python_

Jostar (from the Farsi word “جستار” meaning “finder”) is a Python-based feature selection module comprised of nine different feature selection approaches from single objective to multi-objective methods, for regression and classification tasks. The algorithms, to this date, are:

- Ant Colony Optimization (ACO)
- Differential Evolution (DE)
- Genetic Algorithm (GE)
- Plus-L Minus-R (LRS)
- Non-dominated Sorting Genetic Algorithm (NSGA-II)
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Sequential Forward Selection (SFS)
- Sequential Backward Selection (SBS)

## Features

- User-friendly, Sklearn-like interface (just call "_fit_" )
- Thorough documentation and explanation of hyperparameters and their ranges  
- Tune hyperparameters easily 
- Generate rankings of the selected features  
- Display the results of your classification or regression task

## Example
![](https://doc-14-90-docs.googleusercontent.com/docs/securesc/mhc7nlp7l3ujrv42bt62fmuuq5segrsm/75erp660gkd15qto0onqnjjtcdh6n81o/1624289925000/06678240213073000240/06678240213073000240/1ZP1bZnKhhwLf89tKyHfY3bFkUxiJz2E6?authuser=0)

![](https://doc-0k-90-docs.googleusercontent.com/docs/securesc/mhc7nlp7l3ujrv42bt62fmuuq5segrsm/v3lmfjq6lq780hl0i313qmu75c78296m/1624300650000/06678240213073000240/06678240213073000240/1dnnBAvteXP7EX5z5ECbIN8hPIDuPoggj?authuser=0&nonce=827s3qvf20kq2&user=06678240213073000240&hash=jllof9oeu9aa9i68c57d2njt0mklm5us)


## Installation
Use pip as below to install jostar.

```sh
pip install jostar
```

to test if your installation was successful, change path to the directory and run pytest.

```sh
pytest
```

## Documentation
Jostar comes with a powerful documentation. 

```python
class GA(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_gen=1000, n_pop=20 , cv=None,                                
                 cross_perc = 0.5, mut_perc = 0.3, mut_rate= 0.02, beta = 5,
                 verbose= True, random_state=None,**kwargs):
        """
        Genetic Algorithms or GA is a widely used global optimization algorithm 
        which was first introduced by Holland. GA is based on the natural selection
        in the evolution theory. Properties of GA such as probability of mutation and 
        cross over determines the specifics of the search done in each iteration.
        Additionally, we can also set the proportion of the population we want to
        perform cross over or mutation for. 
                
        Parameters
        ----------
        model : class
            Instantiated Sklearn regression or classification estimator.
        n_f : int
            Number of features needed to be extracted.
        weight : int
            Maximization or minimization objective, for maximization use +1
            and for mimization use -1.
        scoring : callable
            Callable sklearn score or customized score function that takes in y_pred and y_true                                                
        n_gen : int, optional
            Maximum number of generations or iterations. For more complex 
            problems it is better to set this parameter larger. 
            The default is 1000.
        n_pop : int, optional
            Number of population size at each iteration. Typically, this 
            parameter is set to be 10*n_f, but it is dependent on the complexity 
            of the model and it is advised that user tune this parameter based 
            on their problem. The default is 20.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        cross_perc : float, 
            The percentage of the population to perform cross over on. A common 
            choice for this parameter is 0.5. The larger cross_perc is chosen,
            the more exploition of the current population. The default is 0.5.
        mut_perc : float, optional
            The percentage of the population to perform mutation on. This is 
            usually chosen a small percentage (smaller than cross_perc). As 
            mut_perc is set larger, the model explorates more. 
            The default is 0.1.
        mut_rate : float, optional
            The mutation rate. This parameter determines the probability of 
            mutation for each individual in a population. It is often chosen 
            a small number to maintain the current good solutions.
            The default is 0.1.
        beta : int, optional
            Selection Pressure for cross-over. The higher this parameter the 
            stricter the selection of parents for cross-over. This value
            could be an integer [1,10]. The default value
            is 5.        
        verbose : bool, optional
            Wether to print out progress messages. The default is True.
        random_state : int, optional
            Determines the random state for random number generation, 
            cross-validation, and classifier/regression model setting. 
            The default is None.

        Returns
        -------
        Instantiated Optimziation model class.

        """
```

## Developing Jostar
If you would like to develop and run tests for Jostar, run the following command in you virtual environment to install dev dependencies:

```bash
$ pip install -e .[dev]
```

## Acknowledgement adn References
If you found this project useful in your research, consider citing us.
```

```