# Neural Cellular Automata 
An unofficial implementation of the paper [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) in PyTorch.

## Additions and Changes to Model
The goal was to determine if training the model on more than one image every iteration would enable it to generate new images that it hadn't "seen" before.
I used one of the 12 empty layers from the input to add more information needed for the model to learn. This repository can help you understand how to add your additional information in the input layers if needed.

To learn more about the code implementation see [erikhelmut](https://github.com/erikhelmut/neural-cellular-automata/tree/main?tab=readme-ov-file) repository.


### References:
This is a pytorch implementation of the model described in this [paper](https://distill.pub/2020/growing-ca/).

This code is also based on the implementations of the following GitHub users:

[erikhelmut](https://github.com/erikhelmut/neural-cellular-automata/tree/main?tab=readme-ov-file)

[jankrepl](https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata)

[chenmingxiang110](https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata/tree/master)
