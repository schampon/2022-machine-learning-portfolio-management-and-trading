# Machine learning for portfolio management and trading

This material is an introduction to using machine-learning for portfolio management and trading. Given the centrality of programming in hedge funds today, the concepts are exposed using only jupyter notebooks in python. Moreover, we leverage the scikit-learn package to illustrate how machine-learning is used in practice in this context.

## Setup the environment
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    git clone https://github.com/schampon/2022-machine-learning-portfolio-management-and-trading.git
    cd 2022-machine-learning-portfolio-management-and-trading
    
    conda create python=3.9 --name  ml4pmt -c https://conda.anaconda.org/conda-forge/ -y
    conda activate ml4pmt

    pip install -r requirements.txt
    pip install -e . 
    python -m ipykernel install --user --name ml4pmt --display-name "Python (ml4pmt)"
    
    ```
