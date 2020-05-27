# PAL - Parabolic Approximation Line Search for DNNs: 
This repository provides Tensorflow and Pytorch reference implementations for PAL.
PAL is an efficient and effective line search approach for DNNs which exploits the almost parabolic 
shape of the loss in negative gradient direction to automatically estimate good step sizes.

If you have any questions or suggestions, please do not hesitate to contact me: maximus.mutschler(at)uni-tuebingen.de

## Short introduction to PAL:

<img src="/Images/explanation.png " title="PAL' basic idea" alt="PAL' basic idea" width="400"/> 
<!-- d align="right"> Fig1: PAL' basic idea </a> -->

***Fig1: PAL's basic idea*** 

PAL is based on the empirical observarion that the loss function can be approximated by a one-dimensional parabola in negative gradient/line direction.
To do this, one additional point has to be meashured on the line.  
PAL performs a variable update step by jumping into the minimum of the approximated parabola.   
PAL's performance competes ADAM's, SLS's, SGD's and RMSPRPOP's on ResNet-32, MobilenetV2, DenseNet-40 and EfficientNet architectures trained on CIFAR-10 and CIFAR100.  
However, those are tuned by optimal piecewise constant step sizes, whereas PAL does derive its own learning rate schedule.  
Thus, PAL surpasses all those optimizers when they are trained without a schedule.  
Therefore, PAL could be used in scenarios where default schedules fail.  
For a detailed explanation, please refer to our paper.: https://arxiv.org/abs/1903.11991

<p float="left">
<img src="/Images/MOBILENETV2_CIFAR100_train_loss.png" title="Exemplary performance of PAL with data augmentations" alt="Exemplary Performance of PAL with data augmentation" width="300" />
<img src="/Images/MOBILENETV2_CIFAR100_eval_acc.png" title="Exemplary performance of PAL with data augmentations" alt="Exemplary Performance of PAL with data augmentation" width="300" />
<img src="/Images/MOBILENETV2_CIFAR100_step_sizes.png" title="Exemplary performance of PAL with data augmentations" alt="Exemplary Performance of PAL with data augmentation" width="300" />
</p>

***Fig2: Exemplary performance of PAL with data augmentation***

<img src="/Images/ResNetCifarMin30.png" title="Exemplary performance of PAL without data augmentation" alt="Exemplary Performance of PAL of PAL without data augmentation" width="420" />

***Fig3: Exemplary performance of PAL without data augmentation, however this leads to severe overfitting***

## The hyperparameters:

For a detailed explanation, please refer to our paper: https://arxiv.org/abs/1903.11991.  
The introduced hyperparameters lead to good training and test errors:   
 Usually only the measuring step size has to be adapted slightly.
Its sensitivity is not as high as the one of of the learning rate of SGD.  


 <table style="width:100%">
    <tr>
    <th>Abbreviation  </th>
    <th>Name</th>
    <th>Description   </th>
    <th>Default parameter intervalls   </th>
    <th>Sensitivity compared to SGD leaning rate</th>
  </tr>
  <tr>
    <td>&mu; </th>
    <td>measuring step size</th>
    <td>distance to the second sampled training loss value   </th>
    <td>[0.1,1]   </th>
    <td> medium </th>
  </tr>
  <tr>
    <td> &alpha; </td>
    <td>update step adaptation </td>
    <td>Multiplier to the update step </td>
    <td>[1.0,1.2,1.7]   </td>
    <td> low </th>
  </tr>
  <tr>
    <td>&beta;</td>
    <td>conjugate gradient factor </td>
    <td>Adapts the line direction depending on of previous line directions </td>
    <td>[0.0.4] </td>
    <td> low </th>
  </tr>
    <tr>
    <td>s<sub>max</sub> </td>
    <td>maximum step size  </td>
    <td>maximum step size  on line.</td>
     <td>[3.6] </td>
     <td> low </th>
  </tr>
</table> 

## PyTorch Implementation:
- No limitations. Can be used in the same way as any other PyTorch optimizer.
- Runs with PyTorch 1.4
- Uses tensorboardX for plotting
- Parabola approximations and loss lines can be plotted

## Tensorflow Implementation:
- limitations:
    - The DNN must not contain any random components such as Dropout or ShakeDrop. This is because PALS requires two loss values of the same deterministic function (= two network inferences) to determine an update step. Otherwise the function would not be continuous and a parabolic approximation is not be possible. However, if these random component implementations could be changed so that drawn random numbers can be reused for at least two inferences, PAL would also support these operations. 
    - If using Dropout this has to be replaced with the adapted implementation we provide which works with PAL.
    - With Tensorflow 1.15 and 2.0 is was not possible for us to write a completely graph-based optimizer. Therefore it has to be used slightly different as other optimizers. Have a look into the example code! This is not the case with Pytorch.
    - The Tensorflow implementation does not support Keras and Estimator API.
- Runs with Tensorflow 1.15
- Uses tensorboard for plotting
- Parabola approximations and loss lines can be plotted

## Virtual Environment
A virtual environment capable of executing the provided code can be created with the provided python_virtual_env_requirements.txt



