STD=c++23
CC=g++

main: main.cpp board.cpp piece.cpp
	$(CC) -g -o $@ $^ -std=$(STD) -Wall -Wextra

.PHONY: clean
clean:
	rm -rf main