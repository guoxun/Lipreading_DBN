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
model_output_dir=${datafiledir}DBN_Models
data_output_dir=${datafiledir}DBN_Representations

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

echo "Setting up data"
python ConfigData.py ${datafiledir} ${model_output_dir} ${data_output_dir} ${gpu_mem} ${main_mem} 

echo "Computing mean and variance of visual feature"
if [ ! -e  ${datafiledir}/visual_stats.npz ]
then
python ${deepnet}/compute_data_stats.py ${datafiledir}/audiovisualdata.pbtxt ${datafiledir}/visual_stats.npz visual_unlabelled
fi

echo "Computing mean and variance of audio feature"
if [ ! -e  ${datafiledir}/audio_stats.npz ]
then
python ${deepnet}/compute_data_stats.py ${datafiledir}/audiovisualdata.pbtxt ${datafiledir}/audio_stats.npz audio_unlabelled
fi

echo "Training the first visual layer"
if [ ! -e  ${model_output_dir}/visual_rbm1_LAST ]
then
python ${trainer} models/visual_rbm1.pbtxt Trainers/train.pbtxt eval.pbtxt 
fi

echo "Training the second visual layer"
if [ ! -e  ${model_output_dir}/visual_rbm2_rbm_LAST ]
then
python ${trainer} models/visual_rbm2.pbtxt Trainers/train.pbtxt eval.pbtxt
fi

echo "Training the third visual layer"
if [ ! -e  ${model_output_dir}/visual_rbm3_rbm_LAST ]
then
python ${trainer} models/visual_rbm3.pbtxt Trainers/train.pbtxt eval.pbtxt
fi


python ${trainer} models/visual_classifier.pbtxt Trainers/train_classifier.pbtxt eval.pbtxt ${cpu_mem}


echo "Collecting representation from the trained model for the dataset."
if 
python ${extract_rep} ${model_output_dir}/visual_rbm1_LAST Trainers/train_CD_visual_layer1_2.pbtxt visual_hidden1 ${data_output_dir}/visual_rbm1_LAST ${gpu_mem} ${cpu_mem} 
python ${extract_rep} ${model_output_dir}/visual_rbm2_rbm_LAST Trainers/train_CD_visual_layer2.pbtxt visual_hidden2 ${data_output_dir}/visual_rbm2_rbm_LAST ${gpu_mem} ${cpu_mem}
python ${extract_rep} ${model_output_dir}/visual_rbm3_rbm_LAST Trainers/train_CD_visual_layer3.pbtxt visual_hidden3 ${data_output_dir}/visual_rbm3_rbm_LAST ${gpu_mem} ${cpu_mem}
