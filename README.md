# AnomalyDetection4Jets

download files from zenodo
```
kubectl create -f anomaly-pod.yaml
```

log on to pod
```
kubectl exec -it gpu-pod-example  bash
```

set up port forwarding
```
kubectl port-forward gpu-pod-example 8888:8888
```
