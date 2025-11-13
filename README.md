# Confidence Estimation of Qwen VL 4B Instruct 
TODO

## Development notes (to be cleared after release)
- When installing new packages, run this: `pip freeze > requirements.txt`
- If any packages were removed or changed in someone else's commit, run: `pip-sync requirements.txt`
- `Markdown cheat sheet:`https://images.ctfassets.net/wp1lcwdav1p1/4Kz1Ao27PejcyFuWZ93hDy/a1b82cf0db13252cd76763ac529a6285/Screenshot_2024-12-17_at_18.51.54.png

## Install
### Creating the environment
```
conda create -n tf_project python=3.10.18 -y
conda activate tf_project
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
```

## Used Resources
TODO