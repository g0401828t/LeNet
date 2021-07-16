# LeNet
LeNet implemented with pytorch

- main.py에서 parameter 부분의 path (경로)만 본인에 맞게 수정 필요

![R1280x0](https://user-images.githubusercontent.com/55650445/124887962-a064ef00-e010-11eb-85bf-a77102dadd61.png)

### C1
nn.Conv2d(1, 6, k=5, stride=1)

### S2
subsampling
F.avg_pool2d(input, k=2, stride=2)

### C3
nn.Conv2d(6, 16, k=5, stride=1)

### S4
subsampling
F.avg_pool2d(input, k=2, stride=2)

### C5
nn.Conv2d(16, 120, k=5, stride=1)

### F4
fully connected layers
nn.Linear(120, 84)
nn.Linear(84, 10)
F.softmax()
