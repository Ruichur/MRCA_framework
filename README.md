# MRCA Framework: A framework of state estimation for Multi-Robot Cooperative Positioning

## Installation

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install numpy scipy matplotlib

## Getting Started
git clone https://github.com/Ruichur/MRCA_framework.git
cd mrca-framework
python main.py

## Code Structure
mrca-framework/

├── main.py                      # Main execution script

├── framework/                   # Core framework components

│   ├── measurement.py           # Measurement modeling

│   ├── reasoning.py             # Kalman filter implementation

│   ├── cognition.py             # Cooperative mapping

│   └── application.py           # Framework integration

├── DATA/                        # Dataset directory

│   ├── Landmark_Groundtruth.mat

│   ├── Robot1_Groundtruth.mat

│   ├── Robot1_Measurement.mat

│   └── ...

├── results/                     # Output directory

└── README.md

## Configuration
### Trigger mechanism
buffer_num = 15  

### Dataset path
DATA_FOLDER = "/DATA_output7"

### Number of robots and anchors
num_robots = 5
num_anchors = 15

## Citation
If you use MRCA in your research, please cite:

@article{mrca2025,
  title={Toward State Estimation Algorithms in Cooperative Positioning: A Comprehensive Framework Design and Its Realization},
  author={Guangteng Fan, Ruichen Zhang, Xi Chen, Lu Cao},
  journal={IEEE Intelligent Transportation Systems},
  year={2025}
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.
