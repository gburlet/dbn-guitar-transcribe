# Linux settings.
MATLAB_HOME = /usr/local/matlab
MEX         = $(MATLAB_HOME)/bin/mex
CXX         = g++
F77         = gfortran
CFLAGS      = -O3 -fPIC -pthread 
FFLAGS      = -O3 -fPIC -fexceptions 

# Mac OS X settings.
#MATLAB_HOME = /Applications/MATLAB_2013a
#MEX         = $(MATLAB_HOME)/bin/mex
#CXX         = g++
#F77         = gfortran 
#CFLAGS      = -O3 -fPIC -fno-common -fexceptions -no-cpp-precomp 
#FFLAGS      = -O3 -x f77-cpp-input -fPIC -fno-common 

TARGET = lbfgsb

OBJS   = solver.o matlabexception.o matlabscalar.o matlabstring.o   \
         matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
         lbfgsb.o
         
CFLAGS += -Wall -ansi -DMATLAB_MEXFILE

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(MATLAB_HOME)/extern/include -o $@ -c $^

%.o: %.f
	$(F77) $(FFLAGS) -o $@ -c $^

$(TARGET): $(OBJS)
	$(MEX) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) -lgfortran -lm \
        -O -output $@ $^

clean:
	rm -f *.o $(TARGET).mex*

