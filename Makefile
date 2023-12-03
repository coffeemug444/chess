STD=-std=c++23
CC=g++
CFLAGS=-Wall -Wextra

ODIR=obj
LIBS=-lm
SRCDIR=src
BINDIR=bin

CLASSES = board piece
DEPS = $(patsubst %,$(SRCDIR)/%.hpp,$(CLASSES))
OBJ = $(patsubst %,$(ODIR)/%.o,$(CLASSES)) $(ODIR)/main.o

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(STD) $(CFLAGS)
	
main: $(OBJ)
	$(CC) -g -o $(BINDIR)/$@ $^ $(STD) $(CFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -rf $(BINDIR)/main $(ODIR)/*.o
