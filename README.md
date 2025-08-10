# Voxel Path Finder Neural Network

This repository contains code and tools for training and evaluating a neural network for 3D path planning tasks, designed to be used within the **Blender 3D environment**. The project includes dataset generation, inspection utilities, model training scripts, and sample datasets.

> **Note:** This project is intended to be used within Blender 3D, leveraging its environment for data generation and visualization. A special mention for Gautier for the great idea!

## Features

- Synthetic 3D path planning dataset generation (with Blender 3D)
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
- Blender 3D (for dataset generation and visualization)
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

## Inference

Once the training is done, you can use the final model in ./training_outputs/final_model.pth for inference.
There is 2 python files to test inference : 
sampled_model_test.py : used to test the final model generating a path from the sample dataset
randomized_model_test : used to test the final model in a random sample

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

[MIT License](LICENSE) (or specify your license here)

## Acknowledgements

- This project was developed with the assistance of **Claude Code by Anthropic**.
- Inspired by research in 3D path planning and neural networks, and made possible by the Blender 3D community.
- Thanks to all contributors and open-source libraries used in this project.

