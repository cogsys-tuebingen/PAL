# PALS - A fast DNN optimizer based on curvature information
This repository provides Tensorflow and Pytorch reference implementations for PALS .
Note that these optimizations are not fully optimized. This means that more well-engineered implementations could deliver even faster performance.

## Short introduction to PALS:

<img src="/paper/explanation.png " title="PALS' basic idea" alt="PALS' basic idea" width="400"/> 
<!-- d align="right"> Fig1: PALS' basic idea </a> -->

***Fig1: PALS' basic idea*** 


PALS is based on the empirical assumption that the loss function can be approximated by a one-dimensional parabola in negative gradient direction.
PALS performs a variable update step by jumping into the minima of an approximated parabola. To do this, the first and second derivatives in the direction of the negative gradient are calculated using two loss values and one gradient. Then a Newton update step is performed.
PALS matches or exceeds ADAM and SGDM on VGG-16, ResNet-34, ResNET-50 and DenseNet architectures trained on CIFAR-10.
Especially on ResNet architectures PALS shows an excellent performance.
For a detailed explanation, please refer to the original paper.: TODO

<img src="/paper/ResNetCifarMin30.png" title="Performance of PALS" alt="Performance of PALS" width="420" />

***Fig2: Performance of PALS***

## The hyper parameters:

For a detailed explanation, please refer to the original paper: TODO


 <table style="width:100%">
    <tr>
    <th>Abbreviation  </th>
    <th>Name</th>
    <th>Description   </th>
    <th>Default Parameters   </th>
  </tr>
  <tr>
    <td>&mu; </th>
    <td>measuring step size</th>
    <td>distance inbetween the to measured loss values   </th>
    <td>[0.1,0.01]   </th>
  </tr>
  <tr>
    <td> &lambda; </td>
    <td>loose approximation factor </td>
    <td> change of the curvature of the approximated parabola</td>
     <td>0,0.6   </td>
  </tr>
  <tr>
    <td>&alpha;</td>
    <td>momentum  </td>
    <td>usual momentum term </td>
     <td>[0.4,0.6] </td>
  </tr>
    <tr>
    <td>a<sub>max</sub> </td>
    <td>maximum step size  </td>
    <td>maximal step on line. important for stability  </td>
     <td>[0.1,1] </td>
  </tr>
</table> 

We used an epochwise exponential dicay for &lambda; amd a<sub>max</sub>. Good decay rates are: 0.85, 0.95 .
## PALS limitations:
- The DNN must not contain any random components such as Dropout or ShakeDrop. This is because PALS requires two loss values of the same deterministic function (= two network inferences) to determine an update step. Otherwise the function would not be continuous and a parabolic approximation would not be possible. However, if these random component implementations could be changed so that drawn random numbers could be reused for at least two inferences, PALS would also support these operations.
- The PALS update step takes about 1.5 times longer than for ADAM or SGD, but still converges at least as fast.
- With the state of the art of Tensorflow it was not possible for us to write a completely graph-based optimizer. Therefore it cannot be used like other Tensorflow optimizers. This is not the case with Pytorch.
- the Tensorflow implementation does not support Keras and Estimator API.

## Pytorch implementation:
- Runs with PyTorch 1.0
- Uses tensorboardX for plotting
<<<<<<< HEAD
- parabola approximations and loss lines can be plotted
=======


## Tensorflow implementation:
- Runs with Tensorflow 1.12
- Uses tensorboard for plotting
- Has to be used in a slightly different way compared to default Tensorflow optimizers. 
    - Does not support Keras and Estimator API.
    - Iterator outputs must be stored manually in variables. Since two inferences over the same input data are required.
    - Have a look into the example code to see how to use it.
- parabola approximations and loss lines can be plotted





