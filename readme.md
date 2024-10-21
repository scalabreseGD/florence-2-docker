# Florence-2 Serving as Docker Rest Endpoint

## Build and Run
### Version
```shell
export version=1.0.0 
```
### GPU
```shell
docker build . -t florence-2-gpu:${version} -f Dockerfile.gpu --platform linux/amd64
```

```shell
docker run --gpus all -p 8080:8080 -e PORT=8080 florence-2-gpu:${version}
```

### CPU
```shell
docker build --no-cache --progress=plain -t florence-2-cpu:${version} -f Dockerfile.cpu .
```
```shell
docker run -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ florence-2-cpu:${version}
```
