1. Despite good data, if your training loss isn't reducing rapidly, there is something definately wrong with the network/architecture/code.
    - **`DO NOT` use sigmoid in the decoder [as used by the authors]**
    - In the existing solution, sigmoid is used to keep the pixel values between 0 and 1.
    - While the intension is good, it should not have been used because the loss function is MSE and sigmoid + MSE aren't convex


2. **`Upsampling + Conv2D`** v/s **`ConvTranspose2D`** choice depends on the application

3. batch normalization didn't significantly improve the loss. However it is not supposed to help reduce the loss directly. It is supposed to make the loss plot more smoother which it did infact. Need to log it all in the wandb experiments so that it can be compared.