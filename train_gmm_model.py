import matlab.engine

#Hyperparams
K_range = [15, 18]
type = "force" # "pose" or "force"
num_models = 20

#Start matlab
eng = matlab.engine.start_matlab()
for K in range(K_range[0], K_range[1]):
    name = "gmm_drawer_v2_%d" % K
    bll = eng.train_model(name, type, K, num_models)
    print(name, bll)
eng.quit()
