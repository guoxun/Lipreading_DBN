#!/bin/bash

#This script is the main function of performing visual speech recognition using Deep Belief Network (DBN)

# Location of deepnet. EDIT this for your setup.
deepnet=$HOME/library/deepnet/deepnet

#location of the data
datafiledir=$HOME/Lipreading_DBN/

#location of the multimodal learning
multimodaldir=$HOME/Downloads/multimodal_dbn/

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=1G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=16G

trainer=${deepnet}/trainer.py


extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${datafiledir}DBM_Models
data_output_dir=${datafiledir}DBM_Representations

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}



























echo "RBM 1"
python ${trainer} visual_rbm1.pbtxt train.pbtxt eval.pbtxt

echo "RBM 2"
python ${trainer} visual_rbm2.pbtxt train.pbtxt eval.pbtxt

echo "RBM 3"
python ${trainer} visual_rbm3.pbtxt train.pbtxt eval.pbtxt

echo "Classifier"
python ${trainer} visual_classifier.pbtxt train_classifier.pbtxt eval.pbtxt
