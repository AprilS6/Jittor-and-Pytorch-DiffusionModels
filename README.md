## DDPM 复现（Jittor及Pytorch）

### Jittor环境
```
env:
ubuntu22.04 conda3 CUDA11.8 RTX 3090(24GB)*1
python3.10.16-dev
libomp-dev
gcc11.3.0 with libstdc++.so.6-GLIBC_3.4.30
nvcc11.8
pkgs:
jittor=1.3.9
matplotlib=3.10.0
numpy=1.26.4
pillow=11.2.1
tqdm=4.67.1
```

### Pytorch环境
```
env:
ubuntu22.04 conda3 CUDA11.8 RTX 3090(24GB)*1
python3.10.16
pkgs:
```

### More
Pylance类型检查检查不了Jittor！Jittor里面很多动态类型检查，建议关掉Pylance类型检查，不然检查出一堆莫须有的错误傻傻的去各种搜索QAQ。