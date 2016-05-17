rk4_2016_05_13.py

Originator: Tri-Peter Shrive
Contact: tri.shrive@gmail.com
Created: 2016_05_13

Recomended operating system: GNU-Linux. Run in command line with python3.

usage: rk42016_05_13.py [-h] [-A] [-B] [-C] [-D] [a]

Runge - Kutta 4th order method of the Roessler System.

Dx = - y - z
Dy = x + a * y
Dz = b + z * (x - c)

positional arguments:
  a               in Dy = x + a * y

optional arguments:
  -h, --help      show this help message and exit
  -A, --overtime  Plot the System over time.
  -B, --graph3D   Plot a 3D graph of the System.
  -C, --brute     Plot a brute force bifurcation diagram.
  -D, --extrema   Plot a bifurcation diagram using local extrema.

EXAMPLES 
  rk42016_05_13.py -ABCD 0.36
