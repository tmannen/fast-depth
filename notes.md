Run example:

```
python main.py --evaluate ../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar --data sun3d --test_path ../data/bcs_floor6_play_only_formatted/images
```

TODO:

- dataloader gets poses and K? How to take all at once? how to do batch training, how to separate the sequences? maybe write batch dataloader that takes in whole sequence at once? maybe not, but instead just write dataloader to take poses too one at a time and batch size needs to be larger than 1. how to deal with sequences changing between minibatches?
- kernel and distances should be using torch? DONE
- Train with and without kernels? Compare results to fastdepth pretrained on depth? which metrics? maybe test the training first by training without kernels and compare to the pretrained one given by fast depth?
