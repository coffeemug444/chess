STD=-std=c++23
CC=g++
CFLAGS=-Wall -Wextra

ODIR=obj
LIBS=-lOpenCL -lsfml-graphics -lsfml-window -lsfml-system 
SRCDIR=src
BINDIR=bin

CLASSES = board piece mat nnet oclData errors parallelMat
DEPS = $(patsubst %,$(SRCDIR)/%.hpp,$(CLASSES)) 
OBJ = $(patsubst %,$(ODIR)/%.o,$(CLASSES) main)

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(STD) $(CFLAGS)
	
main: $(OBJ)
	$(CC) -g -o $(BINDIR)/$@ $^ $(STD) $(CFLAGS) $(LIBS)
	cp $(SRCDIR)/kernels/*.cl $(BINDIR)/kernels

.PHONY: clean
clean:
	rm -rf $(BINDIR)/main $(ODIR)/*.o $(BINDIR)/kernels/*.cl
