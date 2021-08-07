# Patterner
Patterner is a script which loads in images from a folder, and recreates a target image using patterns. 

It functions by defining a set of positions where every pattern can be placed. Each pattern is placed at each location with a given transparency, making the amount of placement derivable. Code could be extended to not place every pattern at every location. 

In practice the code maps the image to a subspace of the domain of images spanned by the set of patterns at each location. This is a linear optimization problem. The code solves this by using gradient descent. Further, it can be solved using the least squares method of optimizing unsolvable linear optimization problems using the numpy library. Numpy solving only works with small images, as the storage for the basis vectors (patterns) becomes very large quickly. However, if the patterns across several locations do not overlap they can be solved seperately, which works quickly even for larger images. For examples see Examples folder.
