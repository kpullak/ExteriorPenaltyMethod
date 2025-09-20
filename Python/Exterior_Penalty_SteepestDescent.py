import numpy as np
import pandas as pd
import time

def objective_function(x):
    return x[0]**2 + x[1]**2 - 6*x[0] - 8*x[1] + 10

def constraint_functions(x):
    return np.array([4*x[0]**2 + x[1]**2 - 16, 
                     3*x[0] + 5*x[1] - 4, 
                     -x[0], 
                     -x[1]])

def penalty_function(x, mu_value):
    """Penalty function that adds quadratic penalties for violated constraints"""
    f = objective_function(x)
    constraints = constraint_functions(x)
    initial_constraints = constraint_functions([1, 1])  # Check which constraints are violated at starting point
    
    for i in range(4):
        if initial_constraints[i] > 0:
            constraint_at_x = constraint_functions(x)[i]
            f = f + mu_value * constraint_at_x**2
    
    return f

def penalty_gradient(x, mu_value):
    """Gradient of penalty function"""
    # Objective gradient
    obj_grad = np.array([2*x[0] - 6, 2*x[1] - 8])
    
    # Constraint gradients
    constraint_grads = np.array([
        [8*x[0], 2*x[1]],  # gradient of 4*x1^2 + x2^2 - 16
        [3, 5],            # gradient of 3*x1 + 5*x2 - 4
        [-1, 0],           # gradient of -x1
        [0, -1]            # gradient of -x2
    ])
    
    constraints = constraint_functions(x)
    initial_constraints = constraint_functions([1, 1])  # Check which constraints are violated at starting point
    
    penalty_grad = obj_grad.copy()
    for i in range(4):
        if initial_constraints[i] > 0:
            penalty_grad += 2 * mu_value * constraints[i] * constraint_grads[i]
    
    return penalty_grad

# Main code
start_time = time.time()

# Starting point, chosen to violate 1 of the 4 constraints
initial_point = np.array([1.0, 1.0])

# Evaluate constraint values at initial point
constraint_values_at_start = constraint_functions(initial_point)

iteration_number = 1
epsilon = 1e-8
x_new = initial_point.copy()
x_old = initial_point.copy()
mu_value = 10  # Initialization value for mu
results_table = pd.DataFrame()

while mu_value < 1e8:
    mu_value = 10**iteration_number
    
    for counter in range(1, 21):  # 1:20
        if counter != 1 and abs((x_new[0] - x_old[0]) / x_old[0]) < epsilon:
            break
        
        # Calculate gradient
        gradient_f = penalty_gradient(x_old, mu_value)
        search_direction = np.array([-gradient_f[0], -gradient_f[1]])
        
        # Wolfe conditions parameters
        alpha = 1e-4
        beta = 0.9
        step_size = 1/5
        
        x_new = x_old + step_size * search_direction
        function_new = penalty_function(x_new, mu_value)
        gradient_new = penalty_gradient(x_new, mu_value)
        function_old = penalty_function(x_old, mu_value)
        gradient_old = penalty_gradient(x_old, mu_value)
        
        # Wolfe conditions line search
        while ((function_new - function_old) > (alpha * step_size * np.dot(search_direction, gradient_old)) or 
               (np.dot(search_direction, gradient_new) < (beta * np.dot(search_direction, gradient_old)))):
            step_size = step_size / 5
            x_new = x_old + step_size * search_direction
            function_new = penalty_function(x_new, mu_value)
            gradient_new = penalty_gradient(x_new, mu_value)
        
        x_old = x_new.copy()
        x_new = x_new + step_size * search_direction
        
        # Round to 2 decimal places (equivalent to vpa(x,2))
        x_new[0] = round(x_new[0], 2)
        x_new[1] = round(x_new[1], 2)
        function_new = penalty_function(x_new, mu_value)
        
        # Store iteration results
        iteration_data = pd.DataFrame({
            'Iterations': [iteration_number],
            'Mu_Value': [mu_value],
            'x_1': [round(x_new[0], 2)],
            'x_2': [round(x_new[1], 2)],
            'funcValue': [round(function_new, 2)]
        })
        results_table = pd.concat([results_table, iteration_data], ignore_index=True)
    
    iteration_number = iteration_number + 1

print(results_table)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")