# Confidence Estimation of Qwen3 VL 4B Instruct 

***Evaluating original model across the whole dataset***

`src\run_model_on_dataset.py`

***Fine-tuning***

`src\model_fine_tuning.py`

***Evaluating fine-tuned model across the whole dataset***

`src\evaluate_fine_tuned_model.py`

***Analysis of the original and the model fine-tuned for 3 epochs***

`src\analysis_finetuned_3_vs_50_epochs.ipynb`

***Analysis of the model fine-tuned for 3 epochs and for 50 epochs***

`src\analysis_finetuned_3_vs_50_epochs.ipynb`



## Install
### Creating the environment

Using conda:
```
conda create -n tf_project python=3.10.18 -y
conda activate tf_project
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas psutil transformers accelerate pillow pyarrow matplotlib scikit-learn datasets trl
pip install -U bitsandbytes
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip uninstall -y torchao
pip install flash-attn --no-build-isolation
```

If not cuda device available:

`pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu`

In HPC, using venv:
```
mkdir envs
cd envs
module load python/3.10.10
python -m venv tf_project
source tf_project/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas psutil transformers accelerate pillow pyarrow datasets trl
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip uninstall -y torchao
pip install -U bitsandbytes
```

## Used Resources

[1] Yongjin Yang, Haneul Yoo, and Hwaran Lee. MAQA: Evaluating Uncertainty Quantification in
LLMs Regarding Data Uncertainty, March 2025. arXiv:2408.06816 [cs].

[2] Qwen/Qwen3-VL-4B-Instruct · Hugging Face, December 2025. URL https://huggingface.
co/Qwen/Qwen3-VL-4B-Instruct.

[3] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training Verifiers to Solve Math Word Problems, November 2021. arXiv:2110.14168
[cs].

[4] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did
Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies,
January 2021. arXiv:2101.02235 [cs].

[5] Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, and Hang Liu. A Survey of Uncertainty Estimation
Methods on Large Language Models, May 2025. arXiv:2503.00172 [cs].

[6] Abhishek Kumar, Robert Morabito, Sanzhar Umbet, Jad Kabbara, and Ali Emami. Confidence
Under the Hood: An Investigation into the Confidence-Probability Alignment in Large Language
Models, June 2024. arXiv:2405.16282 [cs].

[7] Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco Guzmán, Mark
Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia. Unsupervised Quality Estimation for Neural Machine Translation, July 2020. arXiv:2005.10608 [cs].

[8] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and
Donald Metzler. Confident Adaptive Language Modeling, October 2022. arXiv:2207.07061
[cs].

[9] Unsloth AI - Open Source Fine-tuning & RL for LLMs. URL https://unsloth.ai/.

[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models, October
2021. URL http://arxiv.org/abs/2106.09685. arXiv:2106.09685 [cs].
