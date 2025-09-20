clc;
clear all;

tic
% declaring the variables x1 x2 and mu used in the code 
% (Note: mu is treated as a constant until the last step)
syms x1 x2 mu 

% starting point, chosen in such a way that it violates 1 of the 4
% constraints, i.e., it is outside the feasible boundary
x0 = [1;1];

% evaluating the constraint values at the the initial point x0
k = constraint_Functions(x0);
% returns the constraint functions at point (x1, x2) 
constraints = constraint_Functions([x1,x2]);

f = Objective_Function([x1,x2]);

for i=1:4
    if(k(i)>0) 
        f =  f + mu*constraints(i)^2;      
    end
end

n = 1;
epsilon = 10^-8;
x_new = x0;
x_old = x0;

mu_value = 10; % Initialization value for mu
T = table;
while mu_value < 10^8
    mu_value = 10^n;
    for counter = 1:20   
        if counter ~= 1 && ((abs((x_new(1)-x_old(1))/(x_old(1)))) < epsilon)
            break;
        end
    
        f = subs(f, mu, mu_value);
        grad_f = gradient(f);
        dk(1) = -subs(grad_f(1),{x1,x2},{x_old(1),x_old(2)});
        dk(2) = -subs(grad_f(2),{x1,x2},{x_old(1),x_old(2)});

        alpha = 10^-4;
        neta = 0.9;
        lambda = 1/5;

        x_new = x_old + lambda*dk';

        f_new = subs(f,{x1,x2},{x_new(1),x_new(2)});
        g_new = subs(grad_f,{x1,x2},{x_new(1),x_new(2)});

        f_old = subs(f,{x1,x2},{x_old(1),x_old(2)});
        g_old = subs(grad_f,{x1,x2},{x_old(1),x_old(2)});
    
        while (f_new - f_old) > (alpha*lambda*dk*g_old) || (dk*g_new) < (neta*dk*g_old)
            lambda = lambda/5;
            x_new = x_old + lambda*dk';
            f_new = subs(f,{x1,x2},{x_new(1),x_new(2)});
            g_new = subs(grad_f,{x1,x2},{x_new(1),x_new(2)});
        end
        x_old = x_new;
        x_new = x_new + lambda*dk';
        x_new(1) = vpa(x_new(1),2);
        x_new(2) = vpa(x_new(2),2);
        f_new = subs(f,{x1,x2},{x_new(1),x_new(2)});
        x_t = table(n, mu_value, double(vpa(x_new(1),2)), double(vpa(x_new(2),2)), double(vpa(f_new,2)));
        T = [T; x_t];
    end
    n = n + 1;
end

T.Properties.VariableNames = {'Iterations' 'Mu_Value' 'x_1' 'x_2' 'funcValue'};
disp(T);
toc