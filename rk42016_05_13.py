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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from scipy.signal import argrelextrema

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

	fig = plt.figure(1)
	fig.canvas.set_window_title('')
	fig.suptitle('Roessler System over Time', fontsize=12, fontweight='bold')

	ax = fig.add_subplot(311)
	ax.set_ylabel('Dx')
	plt.plot( vt, vx, 'r' )
	plt.xlim(t0,Tt)

	ay = fig.add_subplot(312)
	ay.set_ylabel('Dy')
	plt.plot( vt, vy, 'r' )
	plt.xlim(t0,Tt)

	az = fig.add_subplot(313)
	az.set_ylabel('Dz')
	az.set_xlabel('Dt')
	plt.plot( vt, vz, 'r' )
	plt.xlim(t0,Tt)

	plt.show()
	return

def graph3d(vx, vy, vz, vt):

	fig = plt.figure()
	fig.canvas.set_window_title('')
	ax = fig.add_subplot(111, projection='3d')
	fig.suptitle('Roessler System in 3D', fontsize=12, fontweight='bold')
	ax.set_ylabel('Dx')
	ax.set_xlabel('Dy')
	ax.set_zlabel('Dz')
	plt.plot( vx, vy, vz, 'r', alpha = 0.5)
	plt.show()
	return

def extrema_worker(a):
	"""worker function for calculating the tragectories of the henon map"""
	vx, vy, vz, vt = rk4(Dx, Dy, Dz, a)

	v = np.array(vz)
#	print(v)
	Lmax = argrelextrema(v, np.greater)
	Lmin = argrelextrema(v, np.less)
	Lext = []
	Lext = np.append(v[Lmin], v[Lmax])
#	print(Lext)
	Lext = Lext.flatten()
	Lext = Lext.tolist()
	return Lext[-10:] 

def extrema_pool(a0, a1, a_n):
	print("\nRunning...\n")
	with mp.Pool(processes=mp.cpu_count()) as pool:
		T = pool.map(extrema_worker, np.arange(a0, a1, a_n))
	print(T)

	fig = plt.figure()
	fig.canvas.set_window_title('')
	fig.suptitle('Bifurcation of the Roessler System', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
#	ax.set_title('')
	ax.set_xlabel('a')
	ax.set_ylabel('Local Extrema of Dz')
	plt.plot(np.arange(a0, a1, a_n), T, 'r.', markersize = 0.65)
	plt.xlim([a0,a1])
	plt.show()

	return

def brute_worker(a):
	"""worker function for calculating the tragectories of the henon map"""
	vx, vy, vz, vt = rk4(Dx, Dy, Dz, a)
	return vy[-20:]

def brute_pool(a0, a1, a_n):
	print("\nRunning...\n")
	with mp.Pool(processes=mp.cpu_count()-1) as pool:
		T = pool.map(brute_worker, np.arange(a0, a1, a_n))
	print(T)
	fig = plt.figure()
	fig.canvas.set_window_title('')
	fig.suptitle('Bifurcation of the Roessler System', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
#	ax.set_title('')
	ax.set_xlabel('a')
	ax.set_ylabel('Transients of Dy')
	plt.plot(np.arange(a0, a1, a_n), T, 'r.', markersize = 0.65)
	plt.xlim([a0,a1])

	plt.show()

	return

def return_values():
	x0, y0, z0 = 1, 0, 0
	b, c = 2, 4

	t0 , Tt = 0, 100
	n =  100 * Tt # RK4 step size

	return x0, y0, z0, b, c, t0, Tt, n

if __name__ == '__main__':
	x0, y0, z0, b, c, t0, Tt, n = return_values()
	a = 0.3606
	a0, a1, a_n = 0.25, 0.4, 0.0001 # a_n - Bifurcation step size

	vx, vy, vz, vt = rk4(Dx, Dy, Dz, a)
#	for x, y, z, t in list(zip(vx, vy, vz, vt))[::1]:
#		print("%1.1f %1.1f %1.1f %1.1f" % (x, y, z, t))


	overtime(vx, vy, vz, vt, t0, Tt)

	graph3d(vx, vy, vz, vt)

#	brute_pool(a0, a1, a_n)

	extrema_pool(a0, a1, a_n)


