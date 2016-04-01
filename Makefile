CC     = mpicc
CFLAGS = -g -O2 -Wall -Wextra -Werror

all: plumber.o

plumber.o : plumber.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f plumber.o
