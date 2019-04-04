# What Parameters Are Collected?

## Distance

<img src="pics/3U3B_resid50.distance.png" alt="distance" class="inline"/>

Darker objects represent shorter distances.

## BB Oritentation

<img src="pics/3U3B_resid50.bb_orientation_angle_rad.png" alt="BB" class="inline"/>

Black represents being close to the backbone of the sphere,
white represents being on the sidechain end of the sphere.

## Distance From Center

<img src="pics/3U3B_resid50.thc.png" alt="Thc" class="inline"/>

This one is complicated, but it is meant to portray the relationship between the ray and the center of the intersecting sphere.
Black represents a smaller value of Thc (in the picture below) and white is a larger value.

![THC](https://www.scratchapixel.com/images/upload/ray-simple-shapes/raysphereisect1.png)

## Chain ID

<img src="pics/3U3B_resid50.chain.png" alt="chain" class="inline"/>

1 if the interesecting sphere is part of the same chain as the source sphere, otherwise -1.
-1 is also returned if the ray does not interest with a sphere,
however that case is treated differently in this picture for the sake of visual clarity. 

The white spheres in the picture are part of the same chain as the source sphere,
the gray spheres are part of a different chain, and
(for the sake of this image)
the background is black.

## Amino Acid Designability

<img src="pics/3U3B_resid50.res_10.png" alt="chain" class="inline"/>

This image is generated for each of the 20 amino acids.
Residues that can adopt that amino acid identity are colored white,
even if they have a different amino acid identity to start out.
A residue position that can adopt all 20 amino acids will be white in every picture.

# Useful Links

[Description of the ray tracing algorithm used](https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection)

[Ray Tracing in 256 lines of C++](https://github.com/ssloy/tinyraytracer)

<!-- For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/). -->
