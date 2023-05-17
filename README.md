# MSANet
### Comparison to other methods on the FLIR datase.
Network	mAP0.5(%)	mAP0.75(%)	mAP0.5:0.95(%)
MSANet	76.2	34	39
![image](https://github.com/ysh-strive/MSANet/assets/25916281/e1aab203-1827-4899-bd9e-91ccc91aa7a7)


### Comparison to other methods on the KAIST datase.
Network	mAP0.5(%)
MSANet	78.6
![image](https://github.com/ysh-strive/MSANet/assets/25916281/580f86c0-46f5-4224-8ab3-1064c18a39c8)

Pytorch Code of our approach for "Multi-scale Aggregation Transformers for Multispectral Object Detection"
### Installation 
For the experimental environment, see requirements.txt.
$ pip install -r requirements.txt


### Train Test and Detect
train: ``` python train.py```

test: ``` python test.py```
