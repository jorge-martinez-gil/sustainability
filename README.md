# 🌱 Sustainability of Machine Learning Models

This repository provides an implementation of a **genetic programming algorithm** to optimize a **multi-objective function**, ensuring the sustainability of machine learning (ML) models. The rationale behind balancing **accuracy, interpretability, and energy efficiency** is to serve as example promoting responsible AI development.

## 🚀 Overview
Genetic Programming (GP) is an evolutionary algorithm inspired by **biological evolution**. It searches for mathematical expressions that optimize predefined objectives. This project applies GP to **evolve ML models** with a focus on three key aspects:

1. **Prediction Accuracy** – Measured by Mean Squared Error (MSE) on a test dataset.
2. **Model Simplicity** – Encouraging interpretable solutions.
3. **Energy Efficiency** – Minimizing computational power consumption.

This approach offers a **practical reference** for sustainable computing and can be adapted to various AI applications.

## 📥 Installation
Ensure you have Python installed, then install the required dependencies using:

```bash
pip install numpy deap scikit-learn
```

## 🔧 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jorge-martinez-gil/sustainability.git
   cd sustainability
   ```

2. **Run the script:**
   ```bash
   python sustainability.py
   ```

The script will:
- Load a sample dataset.
- Split it into training and testing sets.
- Execute the genetic programming algorithm.
- Output the **best evolved models** with their respective performance scores (MSE, simplicity, and energy efficiency).

This implementation provides a **foundation for further improvements**, such as refining the interpretability metric or improving energy consumption calculations.

## 📚 Research & Related Work
This project is based on research into **sustainable semantic similarity models**, integrating accuracy, interpretability, and energy efficiency.

For more details, refer to:

```
@article{martinez2022sustainable,
  title={Sustainable semantic similarity assessment},
  author={Martinez-Gil, Jorge and Chaves-Gonzalez, Jose Manuel},
  journal={Journal of Intelligent & Fuzzy Systems},
  volume={43},
  number={5},
  pages={6163--6174},
  year={2022},
  publisher={IOS Press}
}
```

## 📄 License
This project is licensed under the **MIT License**. See the LICENSE file for details.