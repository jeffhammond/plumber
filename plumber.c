#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <mpi.h>

/********************************************
 * internal data
 ********************************************/

typedef enum {
    BARRIER    = 0,
    BCAST      = 1,
    REDUCE     = 2,
    ALLREDUCE  = 3,
    ALLTOALL   = 4,
    ALLTOALLV  = 5,
    GATHER     = 6,
    ALLGATHER  = 7,
    SCATTER    = 8,
    GATHERV    = 9,
    ALLGATHERV = 10,
    SCATTERV   = 11,
    REDSCAT    = 12,
    REDSCATB   = 13,
    ALLTOALLW  = 14,
    MAX_COLL   = 15
} plumber_collective_t;

uint64_t plumber_collective_count[MAX_COLL];
double   plumber_collective_timer[MAX_COLL];
uint64_t plumber_collective_bytes[MAX_COLL];

/********************************************
 * internal functions
 ********************************************/

static void PLUMBER_init(void)
{
    for (int i=0; i<MAX_COLL; i++) {
        plumber_collective_count[i] = 0;
        plumber_collective_timer[i] = 0.0;
        plumber_collective_bytes[i] = 0;
    }
}

static void PLUMBER_finalize(int collective)
{
}

static size_t PLUMBER_count_dt_to_bytes(int count, MPI_Datatype datatype)
{
    int typesize;
    int rc = PMPI_Type_size(datatype, &typesize);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "PLUMBER: PMPI_Type_size did not succeed\n");
    }
    size_t bytes = (size_t)count * (size_t)typesize;
    return bytes;
}

/* replace with more accurate timer if necessary */
static double PLUMBER_wtime(void)
{
    return PMPI_Wtime();
}

/********************************************
 * MPI wrapper stuff
 ********************************************/

/* initialization and termination */

int MPI_Init(int * argc, char** * argv)
{
    int rc = PMPI_Init(argc, argv);
    PLUMBER_init();
    return rc;
}

int MPI_Init_thread(int * argc, char** * argv, int requested, int * provided)
{
    int rc = PMPI_Init_thread(argc, argv, requested, provided);
    PLUMBER_init();
    return rc;
}

int MPI_Abort(MPI_Comm comm, int errorcode)
{
    PLUMBER_finalize(0); /* noncollective version */
    return PMPI_Abort(comm, errorcode);
}

int MPI_Finalize(void)
{
    PLUMBER_finalize(1); /* collective version */
    return PMPI_Finalize();
}

/* collective communication */

int MPI_Barrier(MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Barrier(comm);
    double t1 = PLUMBER_wtime();
    plumber_collective_count[BARRIER] += 1;
    plumber_collective_timer[BARRIER] += (t1-t0);
    return rc;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Bcast(buffer, count, datatype, root, comm);
    double t1 = PLUMBER_wtime();

    size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

    plumber_collective_count[BCAST] += 1;
    plumber_collective_timer[BCAST] += (t1-t0);
    plumber_collective_bytes[BCAST] += bytes;

    return rc;
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    if (rank==root) {
        bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);
    }

    plumber_collective_count[GATHER] += 1;
    plumber_collective_timer[GATHER] += (t1-t0);
    plumber_collective_bytes[GATHER] += bytes;

    return rc;
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    if (rank==root) {
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
        }
    }

    plumber_collective_count[GATHERV] += 1;
    plumber_collective_timer[GATHERV] += (t1-t0);
    plumber_collective_bytes[GATHERV] += bytes;

    return rc;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = PLUMBER_count_dt_to_bytes(recvcount, recvtype);
    if (rank==root) {
        bytes += size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    }

    plumber_collective_count[SCATTER] += 1;
    plumber_collective_timer[SCATTER] += (t1-t0);
    plumber_collective_bytes[SCATTER] += bytes;

    return rc;
}

int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = PLUMBER_count_dt_to_bytes(recvcount, recvtype);
    if (rank==root) {
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtype);
        }
    }

    plumber_collective_count[SCATTERV] += 1;
    plumber_collective_timer[SCATTERV] += (t1-t0);
    plumber_collective_bytes[SCATTERV] += bytes;

    return rc;
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);

    plumber_collective_count[ALLGATHER] += 1;
    plumber_collective_timer[ALLGATHER] += (t1-t0);
    plumber_collective_bytes[ALLGATHER] += bytes;

    return rc;
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    for (int i=0; i<size; i++) {
        bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
    }

    plumber_collective_count[ALLGATHERV] += 1;
    plumber_collective_timer[ALLGATHERV] += (t1-t0);
    plumber_collective_bytes[ALLGATHERV] += bytes;

    return rc;
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
    bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);

    plumber_collective_count[ALLTOALL] += 1;
    plumber_collective_timer[ALLTOALL] += (t1-t0);
    plumber_collective_bytes[ALLTOALL] += bytes;

    return rc;
}

int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype,
                  void *recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = 0;
    for (int i=0; i<size; i++) {
        bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtype);
        bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
    }

    plumber_collective_count[ALLTOALLV] += 1;
    plumber_collective_timer[ALLTOALLV] += (t1-t0);
    plumber_collective_bytes[ALLTOALLV] += bytes;

    return rc;
}

int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[],
                  void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
    double t1 = PLUMBER_wtime();

    int rank, size;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &size);

    size_t bytes = 0;
    for (int i=0; i<size; i++) {
        bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtypes[i]);
        bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtypes[i]);
    }

    plumber_collective_count[ALLTOALLW] += 1;
    plumber_collective_timer[ALLTOALLW] += (t1-t0);
    plumber_collective_bytes[ALLTOALLW] += bytes;

    return rc;
}

/* point-to-point communication */
