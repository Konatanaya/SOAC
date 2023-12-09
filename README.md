# Supervised Off-policy Actor-Critic (SOAC)

This repository contains the implementation of Supervised Off-policy Actor-Critic (SOAC). The pytorch implementations of SQN, SAC, SQNQ, and SA2C are included as well.

If you use our code or data please cite our paper.

    @inproceedings{wu2023soac,
      title={SOAC: Supervised Off-Policy Actor-Critic for Recommender Systems},
      author={Wu, Shiqing and Xu, Guandong and Wang, Xianzhi},
      booktitle={Proceedings of the 23rd IEEE International Conference on Data Mining},
      pages={1427-1432},
      year={2023}
    }
Please install all necessary packages first using the following command.
    
    pip install -r requirements.txt    

The datasets are compressed due to the Github limitation. Please also extract them first.

You can use the following sample command to run the code.

    --python main.py --rl_model SOAC --rs_model GRU --dataset RC15

Detailed descriptions of all hyperparameters can be found in main.py 
