# 3D Path Planning Neural Network

This repository contains code and tools for training and evaluating a neural network for 3D path planning tasks. The project includes dataset generation, inspection utilities, model training scripts, and sample datasets.

## Features

- Synthetic 3D path planning dataset generation
- Neural network model for pathfinding
- Training and evaluation scripts
- Dataset inspection tools

## Project Structure

```markdown
```
.
├── dataset_inspector.py
├── generate_synthetic.py
├── main_training_script.py
├── pathfinding_nn.py
├── requirements.txt
├── sample_dataset/
│   ├── sample_0000.npz
│   ├── ...
│   └── inspect_npz.py
├── synthetical_builds_training_samples.py
├── training_outputs/
├── training_pipeline.py
```

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd 3d_path_planning_neural_network
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Generate Synthetic Dataset

```sh
python generate_synthetic.py
```

#### Inspect Dataset

```sh
cd sample_dataset
python inspect_npz.py sample_0000.npz
```

#### Train the Neural Network

```sh
python main_training_script.py
```


## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

[MIT License](LICENSE) (or specify your license here)

## Acknowledgements

- Inspired by research in 3D path planning and neural networks, made with the help of Claude by Anthropic.
- Thanks to all contributors and open-source libraries used in this project.

