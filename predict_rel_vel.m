function vel = predict_rel_vel(model_name, x_in)
    % Load required variables
    if isempty(strfind(path, "D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib"))
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib")
    end
    
    load(model_name,'Priors','Mu','Sigma') % loading the model

    % Predict velocity from model
    x_in = x_in.'; %' 
    d = size(x_in, 1);
    fn_handle = @(x) GMR(Priors, Mu, Sigma, x, 1:d, d+1:2*d);
    vel = fn_handle(x_in).';
end