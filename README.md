# Confidence Estimation of Qwen VL 4B Instruct 
TODO

## Development notes (to be cleared after release)
- When installing new packages, change `Creating the environment accordingly`
- `Markdown cheat sheet:`https://images.ctfassets.net/wp1lcwdav1p1/4Kz1Ao27PejcyFuWZ93hDy/a1b82cf0db13252cd76763ac529a6285/Screenshot_2024-12-17_at_18.51.54.png

## Install
### Creating the environment
```
conda create -n tf_project python=3.10.18 -y
conda activate tf_project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install numpy pandas psutil transformers accelerate pillow
pip install -U bitsandbytes
```

## Used Resources
TODO

## Columns
Index dataset categorie subcategorie question right_answer  first_try_max_prob  first_try_entropy   first_try_margin    first_try_verbal    second_try_max_prob  second_try_entropy   second_try_margin    second_try_verbal    first_try_correct   second_try_correct