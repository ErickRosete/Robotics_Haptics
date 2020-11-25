function evaluate_model(model_name, only_sim)
    %% Load packages
    if isempty(strfind(path, "D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib"))
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib")
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\SEDS_lib")
    end
    %% Load demonstrations
    if ~only_sim
        figure
        hold on; grid on
        xlabel('x')
        ylabel('y')
        zlabel('z')
        view(-40,40)
        axis equal
        title('The original demonstrations from the robot.')
    end
    
    clear demos t
    num_demo = 0;  
    for f = 1:50;
        num_demo = num_demo+1;
        data = dlmread(['demonstrations_project/peg_v2/peg_v2_' num2str(f) '.txt']);
        x = data(:, 2:size(data, 2)); % all data
        demos{num_demo} = x';
        t{num_demo} = data(:,1)';

        if ~only_sim
            plot3(x(:,1),x(:,2),x(:,3),'r.')
        end
    end
    %% Preprocessing Data
    tol_cutting = .05; %.005 A threshold on velocity that will be used for trimming demos
    [x0 , xT, Data, index] = preprocess_demos(demos,t,tol_cutting); %preprocessing data
    
    if ~only_sim
        figure
        hold on;grid on
        plot3(Data(1,:),Data(2,:),Data(3,:),'r.')
        d = size(Data,1)/2;
        xlabel('x')
        ylabel('y')
        zlabel('z')
        view(-44,28)
        title('Demonstrations after the preprocessing step.')
        axis equal
        pause(0.1)
    end
    
    %% Simulation
    load(model_name,'Priors','Mu','Sigma') % loading the model

    opt_sim.dt = 0.01;
    opt_sim.i_max = 2000;
    opt_sim.tol = 0.002;
    
    d = size(Data,1)/2; %dimension of data
    x0_all = Data(1:d,index(1:end-1)); %finding initial points of all demonstrations
    fn_handle = @(x) GMR(Priors,Mu,Sigma,x,1:d,d+1:2*d);
    [x xd]=Simulation(x0_all,[],fn_handle,opt_sim); %running the simulator
    
    figure
    hold on;grid on
    plot3(x(1,:),x(2,:),x(3,:),'r.')
    title('Reproductions from the trained model.')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    view(-44,28)
    axis equal