# PAL - A fast DNN optimization method based on curvature information
This repository provides Tensorflow and Pytorch reference implementations for PAL.
Note that these optimizations are not fully optimized. This means that more well-engineered implementations could deliver even faster performance.

If you have any questions or suggestions, please do not hesitate to contact me.: maximus.mutschler(at)uni-tuebingen.de

## Short introduction to PAL:

<img src="/Images/explanation.png " title="PALS' basic idea" alt="PALS' basic idea" width="400"/> 
<!-- d align="right"> Fig1: PALS' basic idea </a> -->

***Fig1: PAL's basic idea*** 


PAL is based on the empirical observarion that the loss function can be approximated by a one-dimensional parabola in negative gradient direction.
PAL performs a variable update step by jumping into the minima of an approximated parabola. To do this, the first and second derivatives in the direction of the negative gradient are calculated using two loss values and one gradient. Then a Newton update step is performed.
PAL's performance matches or exceeds ADAM's and SGD's performance on VGG-16, ResNet-32, ResNET-34 and DenseNet-40 architectures trained on CIFAR-10.
Especially on ResNet architectures PALS shows an excellent performance.
For a detailed explanation, please refer to the our paper.: https://arxiv.org/abs/1903.11991

<img src="/Images/ResNetCifarMin30.png" title="Performance of PALS" alt="Performance of PALS" width="420" />

***Fig2: Performance of PAL***

## The hyper parameters:

For a detailed explanation, please refer to our paper: https://arxiv.org/abs/1903.11991

 <table style="width:100%">
    <tr>
    <th>Abbreviation  </th>
    <th>Name</th>
    <th>Description   </th>
    <th>Default parameter intervalls   </th>
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
    <td>0 or 0.6   </td>
  </tr>
  <tr>
    <td>&alpha;</td>
    <td>momentum  </td>
    <td>usual momentum term </td>
    <td>[0.4,0.6] </td>
  </tr>
    <tr>
    <td>s<sub>max</sub> </td>
    <td>maximum step size  </td>
    <td>maximal step on line. important for stability  </td>
     <td>[0.1,1] </td>
  </tr>
</table> 

We used an epochwise exponential decay for &lambda; and a<sub>max</sub>. Good decay rates are: 0.85, 0.95.
## PAL's limitations:
- The DNN must not contain any random components such as Dropout or ShakeDrop. This is because PALS requires two loss values of the same deterministic function (= two network inferences) to determine an update step. Otherwise the function would not be continuous and a parabolic approximation is not be possible. However, if these random component implementations could be changed so that drawn random numbers could be reused for at least two inferences, PALS would also support these operations.
- The PALS update step takes about 1.5 times longer than for ADAM or SGD, but still converges at least as fast.
- With the state of the art of Tensorflow it was not possible for us to write a completely graph-based optimizer. Therefore it cannot be used like other Tensorflow optimizers. Have a look into the example code! This is not the case with Pytorch.
- the Tensorflow implementation does not support Keras and Estimator API.

## Pytorch implementation:
- Runs with PyTorch 1.0
- Uses tensorboardX for plotting
- parabola approximations and loss lines can be plotted


## Tensorflow implementation:
- Runs with Tensorflow 1.12
- Uses tensorboard for plotting
- Has to be used in a slightly different way compared to default Tensorflow optimizers. 
    - Does not support Keras and Estimator API.
    - Iterator outputs must be stored manually in variables. Since two inferences over the same input data are required.
    - Have a look into the example code to see how to use it.
- parabola approximations and loss lines can be plotted

## Virtual environment
A virtual environment capable of executing the provided code can be created with the provided python_vritual_env_requirements.txt



