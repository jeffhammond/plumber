#include <stdlib.h>

#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i=0; i<100; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Request * reqs = malloc(2*size*sizeof(MPI_Request));
    int * temp = malloc(size*sizeof(int));
    for (int i=0; i<1000; i++) {
        for (int j=0; j<size; j++) {
            MPI_Isend(&rank, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &(reqs[j]));
            MPI_Irecv(&(temp[j]), 1, MPI_INT, j, 0, MPI_COMM_WORLD, &(reqs[size+j]));
        }
        MPI_Waitall(2*size, reqs, MPI_STATUSES_IGNORE);
    }
    free(temp);
    free(reqs);

    for (int i=0; i<100; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
