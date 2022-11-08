# RIANet Implementation

## Installation

1. install carla-engine, leaderboard, and scenario-runner by following the instruction (https://leaderboard.carla.org/get_started/)
2. git clone https://github.com/had1227/carla-rllab.git
3. download https://drive.google.com/uc?id=1rQKFDxGDFi3rBLsMrJzb7oGZvvtwgyiL at carla-rllab/utils/
4. unzip models.zip ; rm models.zip
5. to record log, append following command to .bashrc
"export CARLA_LOG_ROOT={your log folder}"

6. sudo apt-get install libpng16-16
7. install gcc
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

## Run
you need to consider using the right port number

1. run server : . run_server_no_rendering.sh --port={your port number}
2. test : python test_run.py --debug --port={your port number} ----trafficManagerPort={your traffic manager port number}
