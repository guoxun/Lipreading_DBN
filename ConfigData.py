"""----------------------------------------------------------------
Load data from npy and setup the data for DBM learning

 Author--Chao Sui
 University of Western Australia
 Date:28-10-2014

This code is written in light of Nitish Srivastava's code for
multimodal learning using DBM.
http://www.cs.toronto.edu/~nitish/multimodal/index.html
 -----------------------------------------------------------------"""
import os
from deepnet import util
from google.protobuf import text_format
import sys

def SetUpTrainer(data_dir,model_dir,representation_dir):
    trainer_config_names = ['train_CD_visual_layer1.pbtxt',
                            'train_CD_visual_layer2.pbtxt',
                            'train_CD_audio_layer1.pbtxt',
                            'train_CD_audio_layer2.pbtxt',
                            'train_CD_joint_layer.pbtxt']
    for trainer_config_name in trainer_config_names:
        filename=os.path.join('Trainers',trainer_config_name)
        trainer_operation=util.ReadOperation(filename)
        if 'layer1' in trainer_config_name:
            trainer_operation.data_proto_prefix = data_dir
        else:
            trainer_operation.data_proto_prefix = representation_dir
        trainer_operation.checkpoint_directory = model_dir
        with open(filename, 'w') as f:
            text_format.PrintMessage(trainer_operation, f)


def main():
    #Input parameters
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    representation_dir = sys.argv[3]
    avdata_pbtxt_file = os.path.join(data_dir, 'audiovisualdata.pbtxt')
    vdata_pbtxt_file = os.path.join(data_dir, 'visualonlydata.pbtxt')
    gpu_mem = sys.argv[4]
    main_mem = sys.argv[5]

    #Edit the data configuration file
    avdata_pb = util.ReadData(avdata_pbtxt_file)
    avdata_pb.gpu_memory = gpu_mem
    avdata_pb.main_memory = main_mem
    avdata_pb.prefix = data_dir
    with open(avdata_pbtxt_file, 'w') as f:
        text_format.PrintMessage(avdata_pb, f)

    vdata_pb = util.ReadData(vdata_pbtxt_file)
    vdata_pb.gpu_memory = gpu_mem
    vdata_pb.main_memory = main_mem
    vdata_pb.prefix = data_dir
    with open(vdata_pbtxt_file, 'w') as f:
        text_format.PrintMessage(vdata_pb, f)

    #Set up the trainer configuration file
    SetUpTrainer(data_dir,model_dir,representation_dir)

if __name__ == '__main__':
  main()
