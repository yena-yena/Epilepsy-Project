import os
if not os.path.exists("X_total.npy"):
    os.system("python multi_edf_preprocess.py")

if not os.path.exists("eeg_cnn_total.pt"):
    os.system("python train_model_total.py")

os.system("python visualize_confusion.py")
