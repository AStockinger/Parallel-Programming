p4simd.o:	p4simd.h  p4simd.cpp
		g++  -c  p4simd.cpp -o p4simd.o

prog04:	program04.cpp p4simd.o
		g++  -o prog04  program04.cpp p4simd.o  -lm  -fopenmp