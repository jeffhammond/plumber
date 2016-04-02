CC      = mpicc
CFLAGS  = -std=c99
CFLAGS += -Wall -Wextra -Wunused -Wformat
CFLAGS += -Werror

all: plumber.o

plumber.o : plumber.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f plumber.o
