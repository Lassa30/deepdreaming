# Notes on different DeepDream "Tricks"
I am going to implement the tricks one-by-one and log the results in here

Image Pyramid
---
- The results are almost the same as without the Image Pyramid.
- However when more layers are used (~10) with bigger ratio (~0.8) and the initial shape is (224, 224, 3) - the results are a little bit better :)

Random shift
---
- No big difference at all <--(actually there are a difference, I didn't shift the image properly)


KEY
---
- Using VGG16 instead of ResNet is a game changer. The results are much better. But still not good enough in comparison to other deepdreams... (<--- it was before fixing bugs mentioned below)

BUGS
---
- I didn't update the image after doing the pyramid layer -> the result was poor
- It became the real "DeepDream" after fixing this bug
- Not updating the shift: self.horizontal and self.vertical were the same each iteration

Random stuff
---
- play with those settings -- adaptive learning rate, adaptive number of iterations
    ```python
    current_layer = image_pyramid.exponent + 2
    layer_iterations = int(config.num_iter * np.log2(current_layer))
    layer_iterations = max(layer_iterations, config.num_iter)
    learning_rate = config.learning_rate * (1.2 ** -(current_layer - 1))
    print("DEBUG\niterations, learning_rate:", layer_iterations, learning_rate, sep="\n\t")

    optimizer = config.optimizer_class([input_tensor], lr=learning_rate, maximize=True)
    for iteration in range(max(layer_iterations, config.num_iter)):
    ```
- code to see the img and a gradient side by side
    ```python
    print("img and grad shapes")
    print(input_tensor.shape, '\n', input_tensor.grad.data.shape, end='')
    import matplotlib.pyplot as plt
    grad, image = img.proc.to_image(input_tensor), img.proc.to_image(input_tensor.grad.data)
    fig, axis = plt.subplots(1, 3, figsize=(8, 8))
    axis[0].set_title(f"{iteration}, {image.shape}")
    axis[2].hist(grad.flatten(), bins=50)
    grad, image = (grad - grad.min())/(grad.max() - grad.min()), np.clip(image, 0, 1)
    axis = axis.flatten()
    axis[0].imshow(grad)
    axis[1].imshow(image)
    fig.tight_layout()
    print("-"*60)
    ```

TODO
---
- refactor `DeepDream.dream` as `dream` and `dream_guided`
- add tests:
  - deepdream: **(at least try different configurations using hypothesis)**
  - ~~img~~
- add badges: tests from gh actions + test coverage
- local notebooks (github notebooks)
  - trick
  - guided
- kaggle notebooks