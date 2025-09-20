import numpy as np
import pandas as pd

def objective_function(x):
    return x[0]**2 + x[1]**2 - 6*x[0] - 8*x[1] + 10

def constraint_functions(x):
    return np.array([4*x[0]**2 + x[1]**2 - 16, 
                     3*x[0] + 5*x[1] - 4, 
                     -x[0], 
                     -x[1]])

def evaluate_function(x):
    c = 100  # try with 10 & 100 
    return (x[0]-1)**2 + (x[1]-1)**2 + c * (x[0]**2 + x[1]**2 - 0.25)**2

def evaluate_gradient_function(x):
    c = 100  # try with 10 & 100 
    gradient = np.array([2*x[0] - 2 + c*(4*x[0]**3 + 4*x[0]*x[1]**2 - x[0]),
                         2*x[1] - 2 + c*(4*x[1]**3 + 4*x[0]**2*x[1] - x[1])])
    return gradient

# Main optimization code
initial_point = np.array([0.1, 0.1])

# Evaluate constraint values at initial point
constraint_values = constraint_functions(initial_point)

# Check which constraints are violated and build penalty function
penalty_function_terms = []
for i in range(4):
    if constraint_values[i] > 0:
        # This would create penalty terms, but the actual penalty function
        # used in the code is the evaluate_function, not based on constraints
        pass

iteration_counter = 1
results_table = pd.DataFrame()

for counter in range(1, 201):  # 1:200
    mu_value = 10 * iteration_counter
    epsilon = 1e-8
    x_old = initial_point.copy()
    hessian_approx = np.eye(2)  # Initial H (2x2 Identity Matrix)
    step_size = 0.50
    
    search_direction = -hessian_approx @ evaluate_gradient_function(initial_point)
    x_new = initial_point + step_size * search_direction
    function_new = evaluate_function(x_new)
    function_old = evaluate_function(initial_point)
    gradient_direction = -evaluate_gradient_function(initial_point)
    
    while function_new > function_old:  # breaks the loop, when f_new < f
        step_size = step_size / 2
        x_new = initial_point + step_size * gradient_direction
        function_new = evaluate_function(x_new)
    
    x_current = initial_point + step_size * search_direction
    current_hessian = hessian_approx.copy()
    current_point = x_current.copy()
    
    for k in range(1, 1001):  # 1:1000
        gradient_norm = np.linalg.norm(evaluate_gradient_function(current_point))
        function_value = evaluate_function(current_point)
        
        if gradient_norm / (1 + abs(function_value)) < epsilon:
            iteration_data = pd.DataFrame({
                'Iterations': [k], 
                'x1': [current_point[0]], 
                'x2': [current_point[1]], 
                'Condition_Hessian': [np.linalg.cond(current_hessian)], 
                'direction_1': [search_direction[0]], 
                'direction_2': [search_direction[1]], 
                'lambda': [step_size], 
                'FuncValue': [function_value]
            })
            break
        
        iteration_data = pd.DataFrame({
            'Iterations': [k], 
            'x1': [current_point[0]], 
            'x2': [current_point[1]], 
            'Condition_Hessian': [np.linalg.cond(current_hessian)], 
            'direction_1': [search_direction[0]], 
            'direction_2': [search_direction[1]], 
            'lambda': [step_size], 
            'FuncValue': [function_value]
        })
        results_table = pd.concat([results_table, iteration_data], ignore_index=True)
        
        search_direction = -current_hessian @ evaluate_gradient_function(current_point)
        gradient_direction = -evaluate_gradient_function(current_point)
        step_size = 0.50
        current_function = evaluate_function(current_point)
        function_new = evaluate_function(x_new)
        
        while function_new > current_function:  # breaks the loop, when f_new < f
            step_size = step_size / 2
            x_new = initial_point + step_size * gradient_direction
            function_new = evaluate_function(x_new)
        
        x_new = current_point + step_size * search_direction
        step_vector = step_size * search_direction
        gradient_difference = evaluate_gradient_function(x_new) - evaluate_gradient_function(current_point)
        
        # BFGS update
        yk_dot_sk = np.dot(gradient_difference, step_vector)
        if abs(yk_dot_sk) > 1e-12:
            second_term = (np.outer(step_vector - current_hessian @ gradient_difference, gradient_difference) * 
                          np.outer(step_vector, step_vector)) / (yk_dot_sk**2)
            hessian_new = (current_hessian + 
                          (np.outer(step_vector - current_hessian @ gradient_difference, step_vector) + 
                           np.outer(step_vector, step_vector - current_hessian @ gradient_difference)) / yk_dot_sk - 
                          second_term)
            current_hessian = hessian_new
        
        current_point = x_new.copy()
    
    iteration_counter = iteration_counter + 1

results_table = pd.concat([results_table, iteration_data], ignore_index=True)
print(results_table)