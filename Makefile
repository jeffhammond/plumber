CC      = mpicc
CFLAGS  = -std=c99
CFLAGS += -Wall -Wextra -Wunused -Wformat
CFLAGS += -Werror

all: plumber.o

plumber.o : plumber.c
	$(CC) $(CFLAGS) -c $< -o $@

test : test.c plumber.o
	$(CC) $(CFLAGS) $< plumber.o -o $@

check: test
	mpirun -n 4 ./test "bogus" 0 100 "hijinks"

clean:
	-rm -f plumber.o
	-rm -f test
	-rm -rf test.dSYM

