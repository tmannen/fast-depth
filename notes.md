TODO:

- dataloader gets poses and K? How to take all at once? how to do batch training, how to separate the sequences? maybe write batch dataloader that takes in whole sequence at once? maybe not, but instead just write dataloader to take poses too one at a time and batch size needs to be larger than 1. how to deal with sequences changing between minibatches?
- kernel and distances should be using torch? DONE
- 