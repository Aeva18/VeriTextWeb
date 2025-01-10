download_assets:
	gdown --fuzzy 'https://drive.google.com/file/d/11f46tnhG54FWsuoa8FksGoXrjD0hCcf1/view?usp=drive_link'
	gdown https://drive.google.com/drive/folders/1_BAUUrTLphHMgV9oepgi5aUBllu4zNYN?usp=drive_link -O ./ --folder #gpt-neo-125m
	gdown https://drive.google.com/drive/folders/1l-zp_-gHc9VO-1_X-gRWAPxFcdP8aRg8?usp=drive_link -O ./ --folder #gpt2-large

install_req:
	pip install -r requirements.txt