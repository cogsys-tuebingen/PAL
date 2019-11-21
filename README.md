# PAL - Parabolic Approximation Line Search: An efficient and effective line search approach for DNNs
This repository provides Tensorflow and Pytorch reference implementations for PAL.
Note that these implementations
are not fully optimized. This means that more well-engineered implementations could deliver even faster performance.

If you have any questions or suggestions, please do not hesitate to contact me: maximus.mutschler(at)uni-tuebingen.de

## Short introduction to PAL:

<img src="/Images/explanation.png " title="PAL' basic idea" alt="PAL' basic idea" width="400"/> 
<!-- d align="right"> Fig1: PAL' basic idea </a> -->

***Fig1: PAL's basic idea*** 


PAL is based on the empirical observarion that the loss function can be approximated by a one-dimensional parabola in negative gradient/line direction.
To do this, one additional point has to be meashured on the line.
PAL performs a variable update step by jumping into the minimum of the approximated parabola. 
PAL's performance competes ADAM's, RADAM's, SGD's and RMSPRPOP's on ResNet-32, MobilenetV2, DenseNet-40 and EfficientNet architectures trained on CIFAR-10 and CIFAR100.
Despite, PAL has a tendency to decrease the training loss slightly slower as other optimizers, it matches on training loss and test loss. Thus, it generalizes better.
For a detailed explanation, please refer to our paper.: https://arxiv.org/abs/1903.11991

<img src="/Images/EFFICIENTNET_CIFAR10_train_lossMin_time120.png" title="Exemplary performance of PAL with data augmentations" alt="Exemplary Performance of PAL with dataaugmentation" width="420" />
***Fig2: Exemplary performance of PAL with data augmentation***

<img src="/Images/ResNetCifarMin30.png" title="Exemplary performance of PAL without data augmentation" alt="Exemplary Performance of PAL of PAL without data augmentation" width="420" />

***Fig2: Exemplary performance of PAL without data augmentation***

## The hyperparameters:

For a detailed explanation, please refer to our paper: https://arxiv.org/abs/1903.11991.
The introduced hyperparameters lead to good training errors and test errors:


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
    <td>distance to the second sampled training loss value   </th>
    <td>[0.1,1]   </th>
  </tr>
  <tr>
    <td> &alpha; </td>
    <td>update step adaptation </td>
    <td>Multiplier to the update step </td>
    <td>[1.2,1.7]   </td>
  </tr>
  <tr>
    <td>&beta;</td>
    <td>conjugate gradient factor </td>
    <td>Adapts the line direction depending on of previous line directions </td>
    <td>[0.0.4] </td>
  </tr>
    <tr>
    <td>s<sub>max</sub> </td>
    <td>maximum step size  </td>
    <td>maximum step size  on line.</td>
     <td>[1,10] </td>
  </tr>
</table> 

## PAL's limitations:
- The DNN must not contain any random components such as Dropout or ShakeDrop. This is because PALS requires two loss values of the same deterministic function (= two network inferences) to determine an update step. Otherwise the function would not be continuous and a parabolic approximation is not be possible. However, if these random component implementations could be changed so that drawn random numbers can be reused for at least two inferences, PALS would also support these operations. 
- If using Dropout this has to be replaced with the adapted implementation we provide which works with PAL.
- With Tensorflow1.15 is was not possible for us to write a completely graph-based optimizer. Therefore it has to be used slightly different as other optimizers. Have a look into the example code! This is not the case with Pytorch.
- The Tensorflow implementation does not support Keras and Estimator API.

## PyTorch implementation:
- Runs with PyTorch 1.3
- Uses tensorboardX for plotting
- Parabola approximations and loss lines can be plotted


## Tensorflow implementation:
- Runs with Tensorflow 1.13
- Uses tensorboard for plotting
- Has to be used in a slightly different way compared to default Tensorflow optimizers. 
    - Does not support Keras and Estimator API.
    - Iterator outputs must be stored manually in variables. Since two inferences over the same input data are required.
    - Have a look into the example code, to see how to use it.
- Parabola approximations and loss lines can be plotted

## Virtual environment
A virtual environment capable of executing the provided code can be created with the provided python_virtual_env_requirements.txt



