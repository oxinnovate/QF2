# QF2 (Quick Firing learn)

We propose Quick Firing (QF2) Learning, a novel, biologically inspired framework for knowledge consolidation in neural networks. QF2 enables direct, one-shot weight updates via firing-based synaptic rules, without gradient descent or matrix inversion, mimicking engram cell formation and Hebbian learning in the brain. This method is mathematically proven and experimentally validated on large language models, with open-source code and models. QF2 not only advances AI but also offers new inspiration for neuroscience and brain–machine interface research.

## Installation

### Prerequisites
- Python 3.10
- PyTorch 2.3.1+
- CUDA-compatible GPU (recommended)

### Setup
Clone the repository:
```bash
git clone https://github.com/oxinnovate/QF2
cd QF2
```

Install PyTorch with CUDA support:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install accelerate:
```bash
pip install accelerate==1.8.1
```

Install the modified transformers library:
```bash
cd transformers-qf2
pip install .
cd ..
```

Install other dependencies:
```bash
pip install numpy
```


## Usage
### download QFt pre-trained Models
- **Trained Model**: oxinnovate/QF2-1.5B-instruct - Our QF2-trained model


### Basic Usage
Run the main learning script:
```bash
python qf2_learn.py
```



## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{qf_learning_2025,
  title={QF: Quick Feedforward AI Model Training without Gradient Back Propagation},
  author={Feng Qi},
  year={2025},
  cite={https://arxiv.org/abs/2507.04300}
}
@misc{qf2_learning_2025,
  title={QF2: Quick Fine-tuning Framework for Large Language Models},
  author={Feng Qi},
  year={2025},
  url={https://www.preprints.org/manuscript/202507.2318/v1}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
