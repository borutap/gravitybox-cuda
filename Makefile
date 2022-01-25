LDLIBS=-lglut -lGLEW -lGL -lSDL2
all: gravity

utils.o: utils.cu
	nvcc -c utils.cu -Xcudafe --diag_suppress=20012

parameters.o: parameters.cu
	nvcc -c parameters.cu -Xcudafe --diag_suppress=20012

# dzieki fladze dc mozna uzywac funkcji cuda z innego pliku
particle_common.o: particle_common.cu
	nvcc -dc -c particle_common.cu -Xcudafe --diag_suppress=20012

particle_cpu.o: particle_cpu.cu particle_common.o
	nvcc -dc -c particle_cpu.cu -Xcudafe --diag_suppress=20012

particle_gpu.o: particle_gpu.cu particle_common.o
	nvcc -dc -c particle_gpu.cu -Xcudafe --diag_suppress=20012

gravity: gravity.cu particle_gpu.o particle_cpu.o parameters.o utils.o
	nvcc gravity.cu -Xcudafe --diag_suppress=20012 -o gravity particle_gpu.o particle_cpu.o particle_common.o parameters.o utils.o $(LDLIBS)

clean:
	rm -f *.o gravity

.PHONY: all clean