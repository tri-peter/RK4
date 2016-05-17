# -*- coding: utf-8 -*-

""" rk4.py

    Runge - Kutta 4th order method of the Roessler System

    Copyright 2016 Tri-Peter Shrive

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Tri-Peter Shrive
    Anklamer Str 13
    10115 Berlin
    Deutschland

    +49 17 62087558
    Tri.Shrive@gmail.com

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from scipy.signal import argrelmax
from scipy.signal import argrelmin

def rk4(Dx, Dy, Dz, a):
	x0, y0, z0, b, c, t0, Tt, n = return_values()
	vx = [0] * (n + 1)
	vy = [0] * (n + 1)
	vz = [0] * (n + 1)
	vt = [0] * (n + 1)

	h  = (Tt - t0) / float(n)

	vx[0] = x = x0
	vy[0] = y = y0
	vz[0] = z = z0

	for i in range(1, n + 1):
		k1_Dx, k1_Dy, k1_Dz = Dx(y, z), Dy(x, y, a), Dz(x, z, b, c)

		k2_Dx, k2_Dy, k2_Dz = Dx(y + h * 0.5 * k1_Dy, z + h * 0.5 * k1_Dz), Dy(x + h * 0.5 * k1_Dx, y + h * 0.5 * k1_Dy, a), Dz(x + h * 0.5 * k1_Dx, z + h * 0.5 * k1_Dz, b, c)

		k3_Dx, k3_Dy, k3_Dz = Dx(y + h * 0.5 * k2_Dy, z + h * 0.5 * k2_Dz), Dy(x + h * 0.5 * k2_Dx, y + h * 0.5 * k2_Dy, a), Dz(x + h * 0.5 * k2_Dx, z + h * 0.5 * k2_Dz, b, c)

		k4_Dx, k4_Dy, k4_Dz = Dx(y + h * k3_Dy, z + h * k3_Dz), Dy(x + h * k3_Dx, y + h * k3_Dy, a), Dz(x + h * k3_Dx, z + h * k3_Dz, b, c)
		
		vt[i] = t = t0 + i * h
		vx[i] = x = x + h * (k1_Dx + k2_Dx + k2_Dx + k3_Dx + k3_Dx + k4_Dx) / 6
		vy[i] = y = y + h * (k1_Dy + k2_Dy + k2_Dy + k3_Dy + k3_Dy + k4_Dy) / 6
		vz[i] = z = z + h * (k1_Dz + k2_Dz + k2_Dz + k3_Dz + k3_Dz + k4_Dz) / 6

	return vx, vy, vz, vt

def Dx(y, z):
	return - y - z

def Dy(x, y, a):
	return x + a * y

def Dz(x, z, b, c):
	return b + z * (x - c)

def overtime(vx, vy, vz, vt, t0, Tt):

#	for x, y, z, t in list(zip(vx, vy, vz, vt))[::1]:
#		print("%1.1f %1.1f %1.1f %1.1f" % (x, y, z, t))
	n = 2000
	print(len(vx))
	print(len(vt))
	vx, vy, vz, vt = vx[0:n], vy[0:n], vz[0:n], vt[0:n]
	fig = plt.figure(1)
	fig.canvas.set_window_title('')
	fig.suptitle('Roessler System over Time', fontsize=12, fontweight='bold')

	ax = fig.add_subplot(311)
	ax.set_ylabel('x(t)')
	plt.plot( vt, vx, 'r' )

	ay = fig.add_subplot(312)
	ay.set_ylabel('y(t)')
	plt.plot( vt, vy, 'r' )

	az = fig.add_subplot(313)
	az.set_ylabel('z(t)')
	az.set_xlabel('t')
	plt.plot( vt, vz, 'r' )

	plt.show()
	return

def graph3d(vx, vy, vz, vt):

	fig = plt.figure()
	fig.canvas.set_window_title('')
	ax = fig.add_subplot(111, projection='3d')
	fig.suptitle('Roessler System in 3D', fontsize=12, fontweight='bold')
	ax.set_xlabel('x(t)')
	ax.set_ylabel('y(t)')
	ax.set_zlabel('z(t)')
	plt.plot( vx, vy, vz, 'r', alpha = 0.50)
	plt.show()
	return

def extrema_worker(a):
	"""worker function for calculating the local extrema"""
	vx, vy, vz, vt = rk4(Dx, Dy, Dz, a)
	v = vx

	nr_transients = 1000
	v = np.array(v[-nr_transients:])

	Lext = np.zeros(nr_transients)
	Lext[:] = np.NAN

	maxindex = argrelmax(v)
	Lext[maxindex] = v[maxindex]
	
#	minindex = argrelmin(v)
#	Lext[minindex] = v[minindex]	

	return Lext 

def extrema_pool(a0, a1, a_n):
	"""pool function for calculating and plotting the local extrema"""
	print("\nRunning: aStart = %.3e, aStop = %.3e, aStep = %.3e\n" % (a0, a1, a_n))
	



	with mp.Pool(processes=mp.cpu_count()) as pool:
		T = pool.map(extrema_worker, np.arange(a0, a1, a_n))
#	print(T)
#	print(len(T))
#	print((a1-a0)/a_n)

	fig = plt.figure()
	fig.canvas.set_window_title('')
	fig.suptitle('Bifurcation of the Roessler System', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
#	ax.set_title('')
	ax.set_xlabel('a')
	ax.set_ylabel('Local Maxima of y(t)')
	plt.plot(np.arange(a0, a1, a_n),T, 'r.', markersize = 1)
	plt.xlim([a0,a1])
	print("\nComplete.\n")
	plt.show()
	return

def brute_worker(aa):
	"""worker function for calculating the trajectories"""
	vx, vy, vz, vt = rk4(Dx, Dy, Dz, aa)
	return vx[-300:]

def brute_pool(aa0, aa1, aa_n):
	"""pool function for calculating and plotting the tragectories"""
	print("\nRunning: aStart = %.3e, aStop = %.3e, aStep = %.3e\n" % (aa0, aa1, aa_n))

	with mp.Pool(processes=mp.cpu_count()) as pool:
		T = pool.map(brute_worker, np.arange(aa0, aa1, aa_n))
#	print(T)

	fig = plt.figure()
	fig.canvas.set_window_title('')
	fig.suptitle('Bifurcation of the Roessler System', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
#	ax.set_title('')
	ax.set_xlabel('a')
	ax.set_ylabel('Trajectories of y(t)')
	plt.plot(np.arange(aa0, aa1, aa_n), T, 'r.', markersize = 0.65)
	plt.xlim([aa0,aa1])
	print("\nComplete.\n")
	plt.show()

	return

def return_values():
	x0, y0, z0 = 1, 0, 0
	b, c = 2, 3

	t0 , Tt = 0, 3000
	n =  20 * Tt # RK4 step size

	return x0, y0, z0, b, c, t0, Tt, n

if __name__ == '__main__':

	from argparse import RawTextHelpFormatter
	parser = argparse.ArgumentParser(description='Runge - Kutta 4th order method of the Roessler System.\n\nDx = - y - z\nDy = x + a * y\nDz = b + z * (x - c)', epilog='EXAMPLE \n  %(prog)s -ABCD 0.3.606', formatter_class=RawTextHelpFormatter)

	parser.add_argument("a", nargs='?', const=0.4, default=0.4, type=float, help="in Dy = x + a * y")

	parser.add_argument("-A", "--overtime", help="Plot the System over time.", action='store_true')
	parser.add_argument("-B", "--graph3D", help="Plot a 3D graph of the System.", action='store_true')
	parser.add_argument("-C", "--brute", help="Plot a brute force bifurcation diagram.", action='store_true')
	parser.add_argument("-D", "--extrema", help="Plot a bifurcation diagram using local extrema.", action='store_true')

	args = parser.parse_args()

	x0, y0, z0, b, c, t0, Tt, n = return_values()
	a = args.a
	a0, a1, a_n = 0.2, 0.62, 0.001 # extrema, a_n - Bifurcation step size
	aa0, aa1, aa_n = 0.2, 0.64, 0.001 # brute

	vx, vy, vz, vt = rk4(Dx, Dy, Dz, a)

	if args.overtime:
		overtime(vx, vy, vz, vt, t0, Tt)
	if args.graph3D:
		graph3d(vx, vy, vz, vt)
	if args.brute:
		brute_pool(aa0, aa1, aa_n)
	if args.extrema:
		extrema_pool(a0, a1, a_n)


