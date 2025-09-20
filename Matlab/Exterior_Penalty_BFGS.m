clc;
clear all;


% declaring the variables x1 x2 and mu used in the code 
% (Note: mu is treated as a constant until the last step)
syms x1 x2 mu 

% starting point, chosen in such a way that it violates 1 of the 4
% constraints, i.e., it is outside the feasible boundary
x0 = [0.1;0.1];

% evaluating the constraint values at the the initial point x0
k = constraint_Functions(x0);
% returns the constraint functions at point (x1, x2) 
constraints = constraint_Functions([x1,x2]);


for i=1:4
    if(k(i)>0) 
        f =  Objective_Function([x1,x2]) + mu*constraints(i)^2;      
    end
end

n=1;
for counter = 1:200
    mu_value = 10*n;
    epsilon = 10^-8;
    x_old = x0;
    H0 = eye(2,2); % Initial H (2x2 Identity Matrix)
    lambda = 0.50;

    d0 = -H0 * evaluateGradientFunction(f,x0);
    x_new = x0 + lambda*d0;
    f_new = evaluateFunction(f,x_new);
f0 = evaluateFunction(f,x0);

dk = -evaluateGradientFunction(x0);
while(f_new > f0) % breaks the loop, when f_new < f
    lambda = lambda/2;
    x_new = x0 + lambda*dk;
    f_new = evaluateFunction(x_new);
end

x1 = x0 + lambda*d0;
Hk = H0;
x = x1;

T = table;
for k = 1:1000
    if (norm(evaluateGradientFunction(x))/(1+det(evaluateFunction(x))) < epsilon)
        x_t = table(k, x(1), x(2), cond(Hk), d0(1), d0(2), lambda, evaluateFunction(x));
        break;
    end
    
    x_t = table(k, x(1), x(2), cond(Hk), d0(1), d0(2), lambda, evaluateFunction(x));
    T = [T; x_t];
    d0 = -Hk * evaluateGradientFunction(x);
    dk = -evaluateGradientFunction(x);
    lambda = 0.50;
    f = evaluateFunction(x);
    f_new = evaluateFunction(x_new);
    while(f_new > f) % breaks the loop, when f_new < f
        lambda = lambda/2;
        x_new = x0 + lambda * dk;
        f_new = evaluateFunction(x_new);
    end
    x_new = x + lambda * d0;
    sk = lambda * d0;
    yk = evaluateGradientFunction(x_new) - evaluateGradientFunction(x);
    secondTerm = ((sk - Hk*yk).'*yk*(sk*sk.'))/(yk.'*sk)^2;
    H_new = Hk + ((sk - Hk*yk)*sk.' + sk*(sk - Hk*yk).')/(yk.'*sk) - secondTerm;
    Hk = H_new;
    x = x_new;
end
    n=n+1;
end


T = [T; x_t];
T.Properties.VariableNames = {'Iterations' 'x1' 'x2' 'Condition_Hessian' 'di_1' 'di_2' 'lambda' 'FuncValue'};
disp(T);



function f = evaluateFunction(func,x)
    
    c = 100; % try with 10 & 100 
    f = (x(1)-1)^2 + (x(2)-1)^2 + c * (x(1)^2 + x(2)^2 -0.25)^2;
    %f = x(1)^2 + 2*x(2)^2 -2*x(1)*x(2) -2*x(2);
    
end

function f_gradient = evaluateGradientFunction(func, x)
    
    %{
    f_gradient = [2*x(1)-2*x(2);
                  4*x(2)-2*x(1)-2];
    %}
    
    c = 100; % try with 10 & 100 
    f_gradient = [2*x(1) - 2 + c*(4*x(1)^3 + 4*x(1)*x(2)^2 - x(1));
                  2*x(2) - 2 + c*(4*x(2)^3 + 4*x(1)^2*x(2) - x(2))];
              
end