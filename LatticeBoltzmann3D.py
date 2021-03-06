# LatticeBoltzmannDemo.py:  a two-dimensional lattice-Boltzmann "wind tunnel" simulation
# Uses numpy to speed up all array handling.
# Uses matplotlib to plot and animate the curl of the macroscopic velocity field.

# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated data and documentation (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# Except as contained in this notice, the name of the author shall not be used in
# advertising or otherwise to promote the sale, use or other dealings in this
# Software without prior written authorization.

# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).  Other portions of code are based
# on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

# For related materials see http://physics.weber.edu/schroeder/fluids

import numpy, time, matplotlib.pyplot, matplotlib.animation
import mpl_toolkits.mplot3d.axes3d as p3
from mayavi import mlab
from time import sleep

# Define constants:
height = 80							# lattice dimensions
width = 10 #200
depth = 80
viscosity = 0.02					# fluid viscosity
omega = 1 / (3*viscosity + 0.5)		# "relaxation" parameter
u0 = 0.1							# initial and in-flow speed
two9ths = 2.0/9.0
four9ths = 4.0/9.0					# abbreviations for lattice-Boltzmann weight factors
one9th  = 1.0/9.0
one72th = 1.0/72.0
one36th  = 1.0/36.0
performanceData = False				# set to True if performance data is desired

# Initialize all the arrays to steady rightward flow:
n0 = two9ths * (numpy.ones((height,width,depth)) - 1.5*u0**2)	# particle densities along 9 directions
nN = one9th * (numpy.ones((height,width,depth)) - 1.5*u0**2)
nS = one9th * (numpy.ones((height,width,depth)) - 1.5*u0**2)
nE = one9th * (numpy.ones((height,width,depth)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nW = one9th * (numpy.ones((height,width,depth)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nz0 = one9th * (numpy.ones((height,width,depth)) - 1.5*u0**2)
nz1 = one9th * (numpy.ones((height,width,depth)) - 1.5*u0**2)
nNEZ0 = one72th * (numpy.ones((height,width,depth)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSEZ0 = one72th * (numpy.ones((height,width,depth)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNWZ0 = one72th * (numpy.ones((height,width,depth)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSWZ0 = one72th * (numpy.ones((height,width,depth)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNEZ1 = one72th * (numpy.ones((height,width,depth)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSEZ1 = one72th * (numpy.ones((height,width,depth)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNWZ1 = one72th * (numpy.ones((height,width,depth)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSWZ1 = one72th * (numpy.ones((height,width,depth)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)

rho = n0 + nN + nS + nE + nW + nz0 + nz1 +\
		nNEZ0 + nSEZ0 + nNWZ0 + nSWZ0 + nNEZ1 + nSEZ1 + nNWZ1 + nSWZ1		# macroscopic density
ux = (nE + nNEZ0 + nSEZ0 - nW - nNWZ0 - nSWZ0 + nNEZ1 + nSEZ1 - nNWZ1 - nSWZ1) / rho				# macroscopic x velocity
uy = (nN + nNEZ0 + nNWZ0 - nS - nSEZ0 - nSWZ0 + nNEZ1 + nNWZ1 - nSEZ1 - nSWZ1) / rho				# macroscopic y velocity
uz = (nz0 + nNEZ0 + nNWZ0 - nz1 + nSEZ0 + nSWZ0 - nNEZ1 - nNWZ1 - nSEZ1 - nSWZ1) / rho

# Initialize barriers:
barrier = numpy.zeros((height,width,depth), bool)					# True wherever there's a barrier
barrier[(height/2)-8:(height/2)+8, (width/2)-8:(width/2)+8, depth/2] = True			# simple linear barrier
barrierN = numpy.roll(barrier,  1, axis=0)					# sites just north of barriers
barrierS = numpy.roll(barrier, -1, axis=0)					# sites just south of barriers
barrierE = numpy.roll(barrier,  1, axis=1)					# etc.
barrierW = numpy.roll(barrier, -1, axis=1)
barrierZ0 = numpy.roll(barrier,  1, axis=1)
barrierZ1 = numpy.roll(barrier, -1, axis=1)

barrierNE = numpy.roll(barrierN,  1, axis=1)
barrierNW = numpy.roll(barrierN, -1, axis=1)
barrierSE = numpy.roll(barrierS,  1, axis=1)
barrierSW = numpy.roll(barrierS, -1, axis=1)

barrierNEZ0 = numpy.roll(barrierNE,  1, axis=2)
barrierNEZ1 = numpy.roll(barrierNE,  -1, axis=2)

barrierNWZ0 = numpy.roll(barrierNW, 1, axis=2)
barrierNWZ1 = numpy.roll(barrierNW, -1, axis=2)

barrierSEZ0 = numpy.roll(barrierSE, 1, axis=2)
barrierSEZ1 = numpy.roll(barrierSE, -1, axis=2)

barrierSWZ0 = numpy.roll(barrierSW, 1, axis=2)
barrierSWZ1 = numpy.roll(barrierSW, -1, axis=2)


# Move all particles by one step along their directions of motion (pbc):
def stream():
	global n0, nN, nS, nE, nW, nz0, nz1, nNEZ0, nSEZ0, nNWZ0, nSWZ0, nNEZ1, nSEZ1, nNWZ1, nSWZ1
	nN  = numpy.roll(nN,   1, axis=0)	# axis 0 is north-south; + direction is north
	nS  = numpy.roll(nS,  -1, axis=0)
	nE  = numpy.roll(nE,   1, axis=1)	# axis 1 is east-west; + direction is east
	nW  = numpy.roll(nW,  -1, axis=1)
	nz0  = numpy.roll(nz0,  1, axis=2)
	nz1  = numpy.roll(nz1,  -1, axis=2)

	nNEZ0 = numpy.roll(nNEZ0,  1, axis=0)
	nNWZ0 = numpy.roll(nNWZ0,  1, axis=0)
	nNEZ1 = numpy.roll(nNEZ1,  1, axis=0)
	nNWZ1 = numpy.roll(nNWZ1,  1, axis=0)

	nSEZ0 = numpy.roll(nSEZ0, -1, axis=0)
	nSWZ0 = numpy.roll(nSWZ0, -1, axis=0)
	nSEZ1 = numpy.roll(nSEZ1, -1, axis=0)
	nSWZ1 = numpy.roll(nSWZ1, -1, axis=0)

	nNEZ0 = numpy.roll(nNEZ0,  1, axis=1)
	nNWZ0 = numpy.roll(nNWZ0,  1, axis=1)
	nNEZ1 = numpy.roll(nNEZ1,  1, axis=1)
	nNWZ1 = numpy.roll(nNWZ1,  1, axis=1)

	nSEZ0 = numpy.roll(nSEZ0, -1, axis=1)
	nSWZ0 = numpy.roll(nSWZ0, -1, axis=1)
	nSEZ1 = numpy.roll(nSEZ1, -1, axis=1)
	nSWZ1 = numpy.roll(nSWZ1, -1, axis=1)

	nNEZ0 = numpy.roll(nNEZ0,  1, axis=2)
	nNWZ0 = numpy.roll(nNWZ0,  1, axis=2)
	nNEZ1 = numpy.roll(nNEZ1,  1, axis=2)
	nNWZ1 = numpy.roll(nNWZ1,  1, axis=2)

	nSEZ0 = numpy.roll(nSEZ0, -1, axis=2)
	nSWZ0 = numpy.roll(nSWZ0, -1, axis=2)
	nSEZ1 = numpy.roll(nSEZ1, -1, axis=2)
	nSWZ1 = numpy.roll(nSWZ1, -1, axis=2)


	# Use tricky boolean arrays to handle barrier collisions (bounce-back):
	nN[barrierN] = nS[barrier]
	nS[barrierS] = nN[barrier]
	nE[barrierE] = nW[barrier]
	nW[barrierW] = nE[barrier]
	nz0[barrierZ0] = nz1[barrier]
	nz1[barrierZ1] = nz0[barrier]

	nNEZ0[barrierNEZ0] = nSWZ1[barrier]
	nNEZ1[barrierNEZ1] = nSWZ0[barrier]

	nNWZ0[barrierNWZ0] = nSEZ1[barrier]
	nNWZ1[barrierNWZ1] = nSEZ0[barrier]

	nSEZ0[barrierSEZ0] = nNWZ1[barrier]
	nSEZ1[barrierSEZ1] = nNWZ0[barrier]

	nSWZ0[barrierSWZ0] = nNEZ1[barrier]
	nSWZ1[barrierSWZ1] = nNEZ0[barrier]

# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide():
	global n0, nN, nS, nE, nW, nz0, nz1, nNEZ0, nSEZ0, nNWZ0, nSWZ0, nNEZ1, nSEZ1, nNWZ1, nSWZ1, rho, ux, uy, uz
	# rho = n0 + nN + nS + nE + nW + nz0 + nz1 +\
	# 		nNEZ0 + nSEZ0 + nNWZ0 + nSWZ0 + nNEZ1 + nSEZ1 + nNWZ1 + nSWZ1		# macroscopic density
	# ux = (nE + nNEZ0 + nSEZ0 - nW - nNWZ0 - nSWZ0 + nNEZ1 + nSEZ1 - nNWZ1 - nSWZ1) / rho				# macroscopic x velocity
	# uy = (nN + nNEZ0 + nNWZ0 - nS - nSEZ0 - nSWZ0 + nNEZ1 + nNWZ1 - nSEZ1 - nSWZ1) / rho				# macroscopic y velocity
	# uz = (nz0 + nNEZ0 + nNWZ0 - nz1 + nSEZ0 + nSWZ0 - nNEZ1 - nNWZ1 - nSEZ1 - nSWZ1) / rho
	rho = n0 + nN + nS + nE + nW + nz0 + nz1 +\
			nNEZ0 + nSEZ0 + nNWZ0 + nSWZ0 + nNEZ1 + nSEZ1 + nNWZ1 + nSWZ1		# macroscopic density
	ux = (n0 - nN + nz1 - nNEZ0 + nNWZ1 - nSWZ0 + nNEZ1 - nSEZ0 + nSWZ1 - nNWZ0) / (1.0 * rho)				# macroscopic x velocity
	uy = (nS - nE + nz1 - nNEZ0 + nNWZ1 - nSWZ0 + nSEZ0 - nNEZ1 + nNWZ0 - nSWZ1) / (1.0 * rho)			# macroscopic y velocity
	uz = (nW - nz0 + nz1 - nNEZ0 + nSWZ0 - nNWZ1 + nNEZ1 - nSEZ0 + nNWZ0 - nSWZ1) / (1.0 * rho)
	# ux=(n0-nN+nz1-nNEZ0+nNWZ1-nSWZ0+nNEZ1-nSEZ0+nSWZ1-nNWZ0)/rho
	# uy=(nS-nE+nz1-nNEZ0+nNWZ1-nSWZ0+nSEZ0-nNEZ1+nNWZ0-nSWZ1)/rho
	# uz=(nW-nz0+nz1-nNEZ0+nSWZ0-nNWZ1+nNEZ1-nSEZ0+nNWZ0-nSWZ1)/rho
	ux2 = ux * ux				# pre-compute terms used repeatedly...
	uy2 = uy * uy
	uz2 = uz * uz
	u3 = ux2 + uy2 + uz2
	omu215 = 1 - 1.5*u3			# "one minus u2 times 1.5"
	uxuyuz = ux * uy * uz
	square = 1.5*(ux2 + uy2 + uz2)
	#print(square.max())

	eq1 = two9ths * rho * omu215
	eq2 = one9th * rho * (omu215 + 3*uy + 4.5*uy2)
	eq3 = nN - 6.0 * one9th * rho * uy
	eq4 = one9th * rho * (omu215 + 3*ux + 4.5*ux2)
	eq5 = nE - 6.0 * one9th * rho * ux
	eq6 = one9th * rho * (omu215 + 3*uz + 4.5*uz2)
	eq7 = nz0 - 6.0 * one9th * rho * uz

	product = ux+uy+uz
	eq8= one72th * rho * (1.0 + 3*product + 4.5*(product**2) - square)
	eq9 = eq8 - 6.0 * one72th * rho * product
	product = ux+uy-uz
	eq10 = one72th * rho * (1.0 + 3*product + 4.5*(product**2) - square)
	eq11 = eq10 - 6.0 * one72th * rho * product
	product = ux-uy-uz
	eq12 = one72th * rho * (1.0 + 3*product + 4.5*(product**2) - square)
	eq13 = eq12 - 6.0 * one72th * rho * product
	product = ux-uy-uz
	eq14 = one72th * rho * (1.0 + 3*product + 4.5*(product**2) - square)
	eq15 = eq14 - 6.0 * one72th * rho * product

	n0 += eq1 - n0
	nN += eq2 - nN
	nS += eq3 - nS
	nE += eq4 - nE
	nW += eq5 - nW
	nz0 += eq6 - nz0
	nz1 += eq7 - nz1
	nNEZ0 += eq8 - nNEZ0
	nNWZ1 += eq9 - nNWZ1
	nSWZ0 += eq10 - nSWZ0
	nNEZ1 += eq11 - nNEZ1
	nSEZ0 += eq12 - nSEZ0
	nSWZ1 += eq13 - nSWZ1
	nNWZ0 += eq14 - nNWZ0
	nSEZ1 += eq15 - nSEZ1

	# normalize
	n0 = min_max(n0)
	nN = min_max(nN)
	nS = min_max(nS)
	nE = min_max(nE)
	nW = min_max(nW)
	nz0 = min_max(nz0)
	nz1 = min_max(nz1)
	nNEZ0 = min_max(nNEZ0)
	nNWZ1 = min_max(nNWZ1)
	nSWZ0 = min_max(nSWZ0)
	nNEZ1 = min_max(nNEZ1)
	nSEZ0 = min_max(nSEZ0)
	nSWZ1 = min_max(nSWZ1)
	nNWZ0 = min_max(nNWZ0)
	nSEZ1 = min_max(nSEZ1)

	# Force steady rightward flow at ends (no need to set 0, N, and S components):
	nE[:,0] = one9th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nW[:,0] = one9th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNEZ0[:,0] = one72th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSEZ1[:,0] = one72th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSEZ0[:,0] = one72th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNEZ1[:,0] = one72th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNWZ0[:,0] = one72th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSWZ1[:,0] = one72th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSWZ0[:,0] = one72th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNWZ1[:,0] = one72th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)

# # Here comes the graphics and animation...
# theFig = matplotlib.pyplot.figure(figsize=(8,3))
# fluidImage = matplotlib.pyplot.imshow(curl(ux, uy, uz), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1),
# 									cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')
# 		# See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
# bImageArray = numpy.zeros((height, width, depth, 4), numpy.uint8)	# an RGBA image
# bImageArray[barrier,3] = 255								# set alpha=255 only at barrier sites
# barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')

# Attaching 3D axis to the figure
# fig = matplotlib.pyplot.figure()
# ax = p3.Axes3D(fig)
#
# # Setting the axes properties
# ax.set_xlim3d([0.0, 1.0])
# ax.set_xlabel('X')
#
# ax.set_ylim3d([0.0, 1.0])
# ax.set_ylabel('Y')
#
# ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')
#
# ax.set_title('3D Test')
veclen = width * height * depth
xvec = numpy.random.rand(veclen)
yvec = numpy.random.rand(veclen)
zvec = numpy.random.rand(veclen)
svec = numpy.random.rand(veclen)

x = xvec
y = yvec
z = zvec
s = svec
# x = xvec[0:width]
# y = yvec[0:width]
# z = zvec[0:width]
# s = svec[0:width]

# u = numpy.full(width,0.01)
# v = numpy.full(width,0.01)
# w = numpy.full(width,0.01)
#l = mlab.points3d(x, y, z, s, colormap="copper")
#l = mlab.points3d(x, y, z, s, scale_factor=0.03, vmin=0.03)
l = mlab.points3d(x, y, z, s, scale_factor=0.03)
#l = mlab.quiver3d(x, y, z, u, v, w)
#l.glyph.glyph.clamping = False
#l.glyph.color_mode = 'color_by_scalar'
ms = l.mlab_source
#mlab.show()

# Function called for each successive animation frame:
startTime = time.clock()
#frameList = open('frameList.txt','w')		# file containing list of images (to make movie)



def min_max(x, axis=0):
    min = x.min(keepdims=True)
    max = x.max(keepdims=True)
    result = (x-min)/(max-min)
    return result

# Compute curl of the macroscopic velocity field:
def curl(ux, uy, uz):
	global xvec, yvec, zvec, svec
	counter = 0
	tmp = numpy.roll(uy,-1,axis=1) - numpy.roll(uy,1,axis=1) - numpy.roll(ux,-1,axis=0) + numpy.roll(ux,1,axis=0) + numpy.roll(uz,-1,axis=2) - numpy.roll(uz,1,axis=2)

	st = time.clock()
	for y in xrange(height):
		for x in xrange(width):
			for z in xrange(depth):
				yvec[counter] = y/100.0
				xvec[counter] = x/100.0
				zvec[counter] = z/100.0
				svec[counter] = tmp[y][x][z]
				counter += 1
	print("elapse time at3xroop:" + str(time.clock() - st))
	return xvec, yvec, zvec, svec

def nextFrame():							# (arg is the frame number, which we don't need)
	global startTime
	global ms
	if performanceData and (arg%100 == 0) and (arg > 0):
		endTime = time.clock()
		print "%1.1f" % (100/(endTime-startTime)), 'frames per second'
		startTime = endTime
	#frameName = "frame%04d.png" % arg
	#matplotlib.pyplot.savefig(frameName)
	#frameList.write(frameName + '\n')
	for step in range(20):					# adjust number of steps for smooth animation
		stream()
		collide()
	#return curl(ux, uy, uz)

	#ms.scalars = curl(ux, uy, uz)
	x, y, z, s = curl(ux, uy, uz)
	s = min_max(s)
	# for sval in s:
	# 	print(sval)
	# for idx in xrange(height*width):
	# 	print(s[idx])
	ms.trait_set(x=x, y=y, z=z, scalars=s)
	#ms.trait_set(x=x[0:width], y=y[0:width], z=z[0:width], scalars=s[0:width])
	#ms.trait_set(x=numpy.random.rand(width), y=numpy.random.rand(width), z=numpy.random.rand(width), s=numpy.random.rand(width))

	# fluidImage.set_array(curl(ux, uy, uz))
	# return (fluidImage, barrierImage)		# return the figure elements to redraw

while(True):
	nextFrame()
	#randvec = numpy.random.rand(width)
	#ms.trait_set(x=randvec, y=randvec, z=randvec, scalars=randvec)
	#ms.trait_set(x=numpy.random.rand(width), y=numpy.random.rand(width), z=numpy.random.rand(width), scalars=numpy.random.rand(width))

# animate = matplotlib.animation.FuncAnimation(fig, nextFrame, interval=1, blit=True)
# matplotlib.pyplot.show()
