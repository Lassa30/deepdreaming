# Notes on different DeepDream "Tricks"
I am going to implement the tricks one-by-one and log the results in here

Image Pyramid
---
- The results are almost the same as without the Image Pyramid.
- However when more layers are used (~10) with bigger ratio (~0.8) and the initial shape is (224, 224, 3) - the results are a little bit better :)

Random shift
---
- No big difference at all


KEY
---
- Using VGG16 instead of ResNet is a game changer! The results are much better! But still not good enough in comparison to other deepdreams...