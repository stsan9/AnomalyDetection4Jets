# AnomalyDetection4Jets

create pod with custom docker image (has /anomalyvol volume mounted containing data from zenodo)
```
kubectl create -f anomaly-pod.yaml
```

set up port forwarding
```
kubectl port-forward gpu-pod-example 8888:8888
```

log on to pod
```
kubectl exec -it gpu-pod-example  bash
```

once logged on, check out code and launch jupyter
```
git clone https://github.com/stsan9/AnomalyDetection4Jets
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
```