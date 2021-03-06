CC      =  mpicc

# C99 is _required_ to build
CFLAGS  = -std=c99

# obsessive compulsive disorder
CFLAGS += -Wall -Wextra -Wunused -Wformat

# refuse to compile code if there are warnings
CFLAGS += -Werror

# enable memory profiling - not available on Mac
#CFLAGS += -DHAVE_MALLOC_H

# enable thread safety
CFLAGS += -DHAVE_PTHREAD_H

all: plumber.o test attr

plumber.o : plumber.c
	$(CC) $(CFLAGS) -c $< -o $@

test : test.c plumber.o
	$(CC) $(CFLAGS) $< plumber.o -o $@

attr : attr.c
	$(CC) -Wall $< -o $@

check: test
	mpirun -n 4 ./test "bogus" 0 100 "hijinks"

clean:
	-rm -f plumber.o
	-rm -f test
	-rm -f attr
	-rm -rf *.dSYM

