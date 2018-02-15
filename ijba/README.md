


## Resnet-101, scale-224x224  [best scaling strategy]
TAR @ FAR=0.0001 : 0.5276
TAR @ FAR=0.0010 : 0.7609
TAR @ FAR=0.0100 : 0.9148
TAR @ FAR=0.1000 : 0.9845

## Resnet-101, scale-256, center-crop-224
TAR @ FAR=0.0001 : 0.3485
TAR @ FAR=0.0010 : 0.6033
TAR @ FAR=0.0100 : 0.8752
TAR @ FAR=0.1000 : 0.9767

## Resnet-101, scale-224, center-crop-224
TAR @ FAR=0.0001 : 0.4173
TAR @ FAR=0.0010 : 0.6968
TAR @ FAR=0.0100 : 0.9129
TAR @ FAR=0.1000 : 0.9850

---

## Resnet-101-512d-norm, scale-224x224 [cfg-23]
TAR @ FAR=0.0001 : 0.5790
TAR @ FAR=0.0010 : 0.7704
TAR @ FAR=0.0100 : 0.9240
TAR @ FAR=0.1000 : 0.9884

(epoch3)
TAR @ FAR=0.0001 : *0.6100*
TAR @ FAR=0.0010 : *0.7848*
TAR @ FAR=0.0100 : 0.9262
TAR @ FAR=0.1000 : 0.9878

( + sqrt)
TAR @ FAR=0.0001 : 0.6112
TAR @ FAR=0.0010 : 0.7984
TAR @ FAR=0.0100 : 0.9251
TAR @ FAR=0.1000 : 0.9889

( + cosine)
TAR @ FAR=0.0001 : 0.6100
TAR @ FAR=0.0010 : 0.7914
TAR @ FAR=0.0100 : 0.9262
TAR @ FAR=0.1000 : 0.9878

( + cosine  + sqrt)
TAR @ FAR=0.0001 : 0.6067
TAR @ FAR=0.0010 : 0.7987
TAR @ FAR=0.0100 : 0.9248
TAR @ FAR=0.1000 : 0.9885

[cfg-24]


---


## Resnet-101-512d-norm, scale-224x224 [cfg-22, ft stage-2]
TAR @ FAR=0.0001 : 0.5814
TAR @ FAR=0.0010 : 0.7651
TAR @ FAR=0.0100 : 0.9242
TAR @ FAR=0.1000 : 0.9867

---

## Resnet-101-512d, scale-224x224 [cfg-23, ft stage-2]
TAR @ FAR=0.0001 : 0.5926
TAR @ FAR=0.0010 : **0.7919**
TAR @ FAR=0.0100 : 0.9262
TAR @ FAR=0.1000 : 0.9872

## Resnet-101-512d, scale-224x224 [cfg-21, ft stage-1]
TAR @ FAR=0.0001 : 0.5723
TAR @ FAR=0.0010 : 0.7834
TAR @ FAR=0.0100 : 0.9240
TAR @ FAR=0.1000 : 0.9845



