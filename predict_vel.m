function vel = predict_vel(model_name, x0, xT)
    % Load required variables
    if isempty(strfind(path, "D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib"))
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib")
    end
    
    load(model_name,'Priors','Mu','Sigma') % loading the model
    dt = 0.01;

    % Predict velocity from model
    x0 = x0.';
    xT = xT.';
    d = size(x0, 1);
    fn_handle = @(x) GMR(Priors, Mu, Sigma, x, 1:d, d+1:2*d);
    vel = fn_handle(x0 - xT).';
end