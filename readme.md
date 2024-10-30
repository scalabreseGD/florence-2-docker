# Florence-2 Serving as Docker Rest Endpoint

## Before starting
<b>Build the model with GPU requires nvidia base image. Follow the instructions before in order to login into nvcr.io</b> 

<ol>
<li>Be sure you have Docker installed and buildkit available</li> 
<li>Connect to [NVIDIA NGC](https://org.ngc.nvidia.com/setup/personal-keys)</li>
<li>Create a personal key</li>
<li>Login into nvcr.io</li>
</ol>

## Build and Run
### Version
```shell
export version=0.0.4
```
### GPU
```shell
docker build . -t florence-2-gpu:${version} -f Dockerfile.gpu --platform linux/amd64
```

```shell
docker run --gpus all -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ florence-2-gpu:${version}
```

### CPU
```shell
docker build . -t florence-2-cpu:${version} -f Dockerfile.cpu --platform linux/amd64
```
```shell
docker run -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ florence-2-cpu:${version}
```
