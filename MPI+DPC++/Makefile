CXX=mpicxx

#DPCPPROOT=/lus/theta-fs0/projects/DPCPPA100/codeplay-oneapi-for-cuda-ubuntu1804
DPCPPROOT=/grand/IMEXLBM/codeplay/codeplay-oneapi-for-cuda-ubuntu1804
export OMPI_MPICC=clang
export OMPI_MPICXX=clang++

MPICOMP=$(CXX) -I$(DPCPPROOT)/include -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-unknown-cuda-version 

main: System.o lbm.o mpi_exchange.o main.o 
	$(MPICOMP) System.o lbm.o mpi_exchange.o main.o -o main

main.o: main.cpp
	$(MPICOMP) -c main.cpp

System.o: System.cpp
	$(MPICOMP) -c System.cpp

lbm.o: lbm.cpp
	$(MPICOMP) -c lbm.cpp

mpi_exchange.o: System.cpp
	$(MPICOMP) -c exchange.cpp

run: main
	mpirun -np 2 ./main

.PHONY: clean

clean:
	rm -f main *.o
