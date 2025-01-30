#  Sustainability of Machine Learning Models

This project implements a genetic programming algorithm to optimize a multi-objective function leading to ensure the sustainability of ML models.

## 🌟 Introduction
Genetic programming (GP) is an evolutionary algorithm-based methodology inspired by biological evolution to find computer programs that perform a user-defined task. This project uses GP to evolve mathematical expressions that optimize three objectives:
1. Mean Squared Error (MSE) on a test dataset
2. Simplicity of the solution
3. Energy consumed

It can serve as an example for similar projects aiming at more sustainable computing.

## 🛠️ Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using `pip`:

```bash
pip install numpy deap scikit-learn
```

## ⚙️ Usage

1. Clone the repository:

```bash
git clone https://github.com/jorge-martinez-gil/sustainability.git
cd sustainability
```

2. Run the script:

```bash
python sustainability.py
```

The script will load a sample dataset, split it into training and testing sets, and run the genetic programming algorithm. The best solutions found will be printed along with its fitness values (test MSE, simplicity, and energy). This is just an example. To make the solution more realistic, one can look for better ways to calculate the interpretability and also better ways to calculate the energy consumption.


## 📚 Related work

An example of research where we establish the sustainability of semantic similarity models including: accuracy, interpretability, energy consumption:

```
@article{martinez2022sustainable,
  title={Sustainable semantic similarity assessment},
  author={Martinez-Gil, Jorge and Chaves-Gonzalez, Jose Manuel},
  journal={Journal of Intelligent \& Fuzzy Systems},
  volume={43},
  number={5},
  pages={6163--6174},
  year={2022},
  publisher={IOS Press}
}
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
