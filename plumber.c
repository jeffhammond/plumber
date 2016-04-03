#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include <mpi.h>

/********************************************
 * internal data
 ********************************************/

int plumber_profiling_active = 0;
int plumber_sendmatrix_active = 0;

/* capture these in init and use in finalize */
int plumber_argc;
char** plumber_argv;

typedef enum {
    /* sends */
    SEND          = 0,
    BSEND         = 1,
    SSEND         = 2,
    RSEND         = 3,
    ISEND         = 4,
    IBSEND        = 5,
    ISSEND        = 6,
    IRSEND        = 7,
    /* receives */
    RECV          = 8,
    IRECV         = 9,
    MRECV         = 10,
    IMRECV        = 11,
    /* collectives */
    BCAST         = 20,
    REDUCE        = 21,
    ALLREDUCE     = 22,
    ALLTOALL      = 23,
    ALLTOALLV     = 24,
    GATHER        = 25,
    ALLGATHER     = 26,
    SCATTER       = 27,
    GATHERV       = 28,
    ALLGATHERV    = 29,
    SCATTERV      = 30,
    REDSCAT       = 31,
    REDSCATB      = 32,
    ALLTOALLW     = 33,
    /* the end */
    MAX_COMMTYPE  = 34
} plumber_commtype_t;

char plumber_commtype_names[MAX_COMMTYPE][32] = {
"MPI_Send",
"MPI_Bsend",
"MPI_Ssend",
"MPI_Rsend",
"MPI_Isend",
"MPI_Ibsend",
"MPI_Issend",
"MPI_Irsend",
"MPI_Recv",
"MPI_Irecv",
"MPI_Mrecv",
"MPI_Imrecv",
"MPI_Bcast",
"MPI_Reduce",
"MPI_Allreduce",
"MPI_Alltoall",
"MPI_Alltoallv",
"MPI_Gather",
"MPI_Allgather",
"MPI_Scatter",
"MPI_Gatherv",
"MPI_Allgatherv",
"MPI_Scatterv",
"MPI_Reduce_scatter",
"MPI_Reduce_scatter_block",
"MPI_Alltoallw"
};

typedef enum {
    /* request completion */
    WAIT          = 0,
    WAITANY       = 1,
    WAITSOME      = 2,
    WAITALL       = 3,
    TEST          = 4,
    TESTANY       = 5,
    TESTSOME      = 6,
    TESTALL       = 7,
    /* collectives */
    BARRIER       = 8,
    COMMDUP       = 9,
    COMMCREATE    = 10,
    COMMSPLIT     = 11,
    COMMFREE      = 12,
    /* the end */
    MAX_UTILTYPE  = 13
} plumber_utiltype_t;

char plumber_utiltype_names[MAX_UTILTYPE][32] = {
"MPI_Wait",
"MPI_Waitany",
"MPI_Waitsome",
"MPI_Waitall",
"MPI_Test",
"MPI_Testany",
"MPI_Testsome",
"MPI_Testall",
"MPI_Barrier",
"MPI_Comm_dup",
"MPI_Comm_create",
"MPI_Comm_split",
"MPI_Comm_free",
};

uint64_t plumber_commtype_count[MAX_COMMTYPE];
double   plumber_commtype_timer[MAX_COMMTYPE];
uint64_t plumber_commtype_bytes[MAX_COMMTYPE];

uint64_t plumber_utiltype_count[MAX_UTILTYPE];
double   plumber_utiltype_timer[MAX_UTILTYPE];

/* dynamically allocated due to O(nproc) */
uint64_t * plumber_sendmatrix_count;
double   * plumber_sendmatrix_timer;
uint64_t * plumber_sendmatrix_bytes;

double plumber_start_time;

/********************************************
 * internal functions
 ********************************************/

/* replace with more accurate timer if necessary */
static double PLUMBER_wtime(void)
{
    return PMPI_Wtime();
}

static void PLUMBER_init(int argc, char** argv)
{
    plumber_profiling_active = 1;

    if (plumber_profiling_active) {

        plumber_argc = argc;
        plumber_argv = argv;

        for (int i=0; i<MAX_COMMTYPE; i++) {
            plumber_commtype_count[i] = 0;
            plumber_commtype_timer[i] = 0.0;
            plumber_commtype_bytes[i] = 0;
        }

        for (int i=0; i<MAX_UTILTYPE; i++) {
            plumber_utiltype_count[i] = 0;
            plumber_utiltype_timer[i] = 0.0;
        }

        plumber_sendmatrix_active = 1;

        if (plumber_sendmatrix_active) {
            int size;
            PMPI_Comm_size(MPI_COMM_WORLD, &size);

            plumber_sendmatrix_count = malloc(size * sizeof(uint64_t));
            plumber_sendmatrix_timer = malloc(size * sizeof(double));
            plumber_sendmatrix_bytes = malloc(size * sizeof(uint64_t));

            if (plumber_sendmatrix_count == NULL || plumber_sendmatrix_timer == NULL || plumber_sendmatrix_bytes == NULL) {
                fprintf(stderr, "PLUMBER: sendmatrix memory allocation did not succeed for %d processes\n", size);
                PMPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i=0; i<size; i++) {
                plumber_sendmatrix_count[i] = 0;
                plumber_sendmatrix_timer[i] = 0.0;
                plumber_sendmatrix_bytes[i] = 0;
            }

        }

        plumber_start_time = PLUMBER_wtime();
    }
}

static void PLUMBER_finalize(int collective)
{
    if (plumber_profiling_active) {

        double plumber_end_time = PLUMBER_wtime();
        double plumber_app_time = plumber_end_time - plumber_start_time;

        int rank, size;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);

        char summaryfilepath[255];
        char rankfilepath[255];
        char matrixfilepath[255];

        char * prefix = getenv("PLUMBER_PREFIX");
        if (prefix != NULL) {
            strncpy(summaryfilepath, prefix, 255);
            strncpy(rankfilepath,    prefix, 255);
            strncpy(matrixfilepath,  prefix, 255);
        } else {
            char plumber_program_name[255];
            if (plumber_argc>0) {
                strncpy(plumber_program_name, plumber_argv[0], 255);
            } else {
                strncpy(plumber_program_name, "unknown", 255);
            }
            /* append plumber_program_name with timestamp to be unique... */
            strncpy(summaryfilepath, plumber_program_name, 255);
            strncpy(rankfilepath,    plumber_program_name, 255);
            strncpy(matrixfilepath,  plumber_program_name, 255);
        }

        /* 2^31 = 2147483648 requires 10 digits */
        char rankstring[12] = {0};
        sprintf(rankstring, "%d", rank);

        strcat(summaryfilepath, ".plumber.summary.");
        strcat(summaryfilepath, rankstring);

        strcat(rankfilepath, ".plumber.profile.");
        strcat(rankfilepath, rankstring);

        strcat(matrixfilepath, ".plumber.matrix.");
        strcat(matrixfilepath, rankstring);

        /* this will blast existing files with the same name.
         * we should test for this case and not overwrite them. */
        FILE * rankfile = fopen(rankfilepath, "w");
        if ( rankfile==NULL ) {
            fprintf(stderr, "PLUMBER: fopen of rankfile %s did not succeed\n", rankfilepath);
        } else {
            fprintf(rankfile, "PLUMBER profile for process %d\n", rank);
            /* process name */
            char procname[MPI_MAX_PROCESSOR_NAME] = {0};
            int len;
            MPI_Get_processor_name(procname, &len);
            fprintf(rankfile, "MPI_Get_processor_name = %s\n", procname);
            /* application stats */
            fprintf(rankfile, "total application time = %lf\n", plumber_app_time);
            double plumber_total_mpi_time = 0.0;
            for (int i=0; i<MAX_COMMTYPE; i++) {
                plumber_total_mpi_time += plumber_commtype_timer[i];
            }
            for (int i=0; i<MAX_UTILTYPE; i++) {
                plumber_total_mpi_time += plumber_utiltype_timer[i];
            }
            fprintf(rankfile, "total MPI time = %lf (%6.2lf %)\n",
                              plumber_total_mpi_time,
                              100*plumber_total_mpi_time/plumber_app_time);
            /* MPI profile */
            fprintf(rankfile, "%32s %20s %30s %20s\n", "function", "calls", "time", "bytes");
            for (int i=0; i<MAX_COMMTYPE; i++) {
                if (plumber_commtype_count[i] > 0) {
                    fprintf(rankfile, "%32s %20llu %30.14lf %20llu\n",
                            plumber_commtype_names[i],
                            plumber_commtype_count[i],
                            plumber_commtype_timer[i],
                            plumber_commtype_bytes[i]);
                }
            }
            for (int i=0; i<MAX_UTILTYPE; i++) {
                if (plumber_utiltype_count[i] > 0) {
                    fprintf(rankfile, "%32s %20llu %30.14lf\n",
                            plumber_utiltype_names[i],
                            plumber_utiltype_count[i],
                            plumber_utiltype_timer[i]);
                }
            }
            fprintf(rankfile, "EOF");
            fclose(rankfile);
        }

        if (plumber_sendmatrix_active) {
            FILE * matrixfile = fopen(matrixfilepath, "w");
            if ( matrixfile==NULL ) {
                fprintf(stderr, "PLUMBER: fopen of matrixfile %s did not succeed\n", matrixfilepath);
            } else {
                fprintf(matrixfile, "PLUMBER matrix for process %d\n", rank);
                fprintf(matrixfile, "%10s %10s %30s %20s\n", "target", "calls", "time", "bytes");
                for (int i=0; i<size; i++) {
                    if (plumber_sendmatrix_count[i] > 0) {
                        fprintf(rankfile, "%10d %20llu %30.14lf %20llu\n",
                                i,
                                plumber_sendmatrix_count[i],
                                plumber_sendmatrix_timer[i],
                                plumber_sendmatrix_bytes[i]);
                    }
                }
                fprintf(matrixfile, "EOF");
                fclose(matrixfile);
            }

            free(plumber_sendmatrix_count);
            free(plumber_sendmatrix_timer);
            free(plumber_sendmatrix_bytes);
        }

        if (collective) {
            /* reduce to get totals */
            uint64_t total_commtype_count[MAX_COMMTYPE];
            double   total_commtype_timer[MAX_COMMTYPE];
            uint64_t total_commtype_bytes[MAX_COMMTYPE];
            uint64_t total_utiltype_count[MAX_UTILTYPE];
            double   total_utiltype_timer[MAX_UTILTYPE];
            for (int i=0; i<MAX_COMMTYPE; i++) {
                PMPI_Reduce(&(plumber_commtype_count[i]), &(total_commtype_count[i]),
                            1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
                PMPI_Reduce(&(plumber_commtype_timer[i]), &(total_commtype_timer[i]),
                            1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                PMPI_Reduce(&(plumber_commtype_bytes[i]), &(total_commtype_bytes[i]),
                            1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            for (int i=0; i<MAX_UTILTYPE; i++) {
                PMPI_Reduce(&(plumber_utiltype_count[i]), &(total_utiltype_count[i]),
                            1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
                PMPI_Reduce(&(plumber_utiltype_timer[i]), &(total_utiltype_timer[i]),
                            1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }

            /* write file from rank 0 */
            if (rank==0) {
                FILE * summaryfile = fopen(summaryfilepath, "w");
                if ( summaryfile==NULL ) {
                    fprintf(stderr, "PLUMBER: fopen of summaryfile %s did not succeed\n", summaryfilepath);
                } else {
                    fprintf(summaryfile, "PLUMBER summary\n");
                    /* program invocation */
                    fprintf(summaryfile, "program invocation was:");
                    for (int i=0; i<plumber_argc; i++) {
                        fprintf(summaryfile, " %s", plumber_argv[i]);
                    }
                    fprintf(summaryfile, "\n");
                    /* aggregrate MPI profile */
                    fprintf(rankfile, "%32s %20s %30s %20s\n", "function", "calls", "time", "bytes");
                    for (int i=0; i<MAX_COMMTYPE; i++) {
                        if (total_commtype_count[i] > 0) {
                            fprintf(rankfile, "%32s %20llu %30.14lf %20llu\n",
                                    plumber_commtype_names[i],
                                    total_commtype_count[i],
                                    total_commtype_timer[i],
                                    total_commtype_bytes[i]);
                        }
                    }
                    for (int i=0; i<MAX_UTILTYPE; i++) {
                        if (total_utiltype_count[i] > 0) {
                            fprintf(rankfile, "%32s %20llu %30.14lf\n",
                                    plumber_utiltype_names[i],
                                    total_utiltype_count[i],
                                    total_utiltype_timer[i]);
                        }
                    }
                    fprintf(summaryfile, "EOF");
                    fclose(summaryfile);
                }
            }
        }

    }
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

/********************************************
 * MPI wrapper stuff
 ********************************************/

/* initialization and termination */

int MPI_Init(int * argc, char** * argv)
{
    int rc = PMPI_Init(argc, argv);
    if (argc != NULL && argv != NULL) {
        PLUMBER_init(*argc, *argv);
    } else {
        PLUMBER_init(0,NULL);
    }
    return rc;
}

int MPI_Init_thread(int * argc, char** * argv, int requested, int * provided)
{
    int rc = PMPI_Init_thread(argc, argv, requested, provided);
    if (argc != NULL && argv != NULL) {
        PLUMBER_init(*argc, *argv);
    } else {
        PLUMBER_init(0,NULL);
    }
    return rc;
}

int MPI_Abort(MPI_Comm comm, int errorcode)
{
    /* need to setup error handler to dump logs from all ranks
     * when job is aborted... */
    PLUMBER_finalize(0); /* noncollective version */
    return PMPI_Abort(comm, errorcode);
}

int MPI_Finalize(void)
{
    PLUMBER_finalize(1); /* collective version */
    return PMPI_Finalize();
}

/* collective utility functions */

int MPI_Barrier(MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Barrier(comm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[BARRIER] += 1;
        plumber_utiltype_timer[BARRIER] += (t1-t0);
    }
    return rc;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_dup(comm, newcomm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[COMMDUP] += 1;
        plumber_utiltype_timer[COMMDUP] += (t1-t0);
    }
    return rc;
}

int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_dup_with_info(comm, info, newcomm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[COMMDUP] += 1;
        plumber_utiltype_timer[COMMDUP] += (t1-t0);
    }
    return rc;
}

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_create(comm, group, newcomm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[COMMCREATE] += 1;
        plumber_utiltype_timer[COMMCREATE] += (t1-t0);
    }
    return rc;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_split(comm, color, key, newcomm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[COMMSPLIT] += 1;
        plumber_utiltype_timer[COMMSPLIT] += (t1-t0);
    }
    return rc;
}

int MPI_Comm_free(MPI_Comm *comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_free(comm);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_count[COMMFREE] += 1;
        plumber_utiltype_timer[COMMFREE] += (t1-t0);
    }
    return rc;
}

/* collective communication */

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int size;
        PMPI_Comm_size(comm, &size);

        /* i have no idea if this definition of bytes makes sense */
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[REDUCE] += 1;
        plumber_commtype_timer[REDUCE] += (t1-t0);
        plumber_commtype_bytes[REDUCE] += bytes;
    }

    return rc;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int size;
        PMPI_Comm_size(comm, &size);

        /* i have no idea if this definition of bytes makes sense */
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[ALLREDUCE] += 1;
        plumber_commtype_timer[ALLREDUCE] += (t1-t0);
        plumber_commtype_bytes[ALLREDUCE] += bytes;
    }

    return rc;
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int size;
        PMPI_Comm_size(comm, &size);

        /* i have no idea if this definition of bytes makes sense */
        size_t bytes = 0;
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], datatype);
        }

        plumber_commtype_count[REDSCAT] += 1;
        plumber_commtype_timer[REDSCAT] += (t1-t0);
        plumber_commtype_bytes[REDSCAT] += bytes;
    }

    return rc;
}

int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int size;
        PMPI_Comm_size(comm, &size);

        /* i have no idea if this definition of bytes makes sense */
        size_t bytes = size * PLUMBER_count_dt_to_bytes(recvcount, datatype);

        plumber_commtype_count[REDSCATB] += 1;
        plumber_commtype_timer[REDSCATB] += (t1-t0);
        plumber_commtype_bytes[REDSCATB] += bytes;
    }

    return rc;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Bcast(buffer, count, datatype, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[BCAST] += 1;
        plumber_commtype_timer[BCAST] += (t1-t0);
        plumber_commtype_bytes[BCAST] += bytes;
    }
    return rc;
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        if (rank==root) {
            bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);
        }

        plumber_commtype_count[GATHER] += 1;
        plumber_commtype_timer[GATHER] += (t1-t0);
        plumber_commtype_bytes[GATHER] += bytes;
    }

    return rc;
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        if (rank==root) {
            for (int i=0; i<size; i++) {
                bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
            }
        }

        plumber_commtype_count[GATHERV] += 1;
        plumber_commtype_timer[GATHERV] += (t1-t0);
        plumber_commtype_bytes[GATHERV] += bytes;
    }

    return rc;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = PLUMBER_count_dt_to_bytes(recvcount, recvtype);
        if (rank==root) {
            bytes += size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        }

        plumber_commtype_count[SCATTER] += 1;
        plumber_commtype_timer[SCATTER] += (t1-t0);
        plumber_commtype_bytes[SCATTER] += bytes;
    }

    return rc;
}

int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = PLUMBER_count_dt_to_bytes(recvcount, recvtype);
        if (rank==root) {
            for (int i=0; i<size; i++) {
                bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtype);
            }
        }

        plumber_commtype_count[SCATTERV] += 1;
        plumber_commtype_timer[SCATTERV] += (t1-t0);
        plumber_commtype_bytes[SCATTERV] += bytes;
    }

    return rc;
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);

        plumber_commtype_count[ALLGATHER] += 1;
        plumber_commtype_timer[ALLGATHER] += (t1-t0);
        plumber_commtype_bytes[ALLGATHER] += bytes;
    }

    return rc;
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
        }

        plumber_commtype_count[ALLGATHERV] += 1;
        plumber_commtype_timer[ALLGATHERV] += (t1-t0);
        plumber_commtype_bytes[ALLGATHERV] += bytes;
    }

    return rc;
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = size * PLUMBER_count_dt_to_bytes(sendcount, sendtype);
        bytes += size * PLUMBER_count_dt_to_bytes(recvcount, recvtype);

        plumber_commtype_count[ALLTOALL] += 1;
        plumber_commtype_timer[ALLTOALL] += (t1-t0);
        plumber_commtype_bytes[ALLTOALL] += bytes;
    }

    return rc;
}

int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype,
                  void *recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = 0;
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtype);
            bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtype);
        }

        plumber_commtype_count[ALLTOALLV] += 1;
        plumber_commtype_timer[ALLTOALLV] += (t1-t0);
        plumber_commtype_bytes[ALLTOALLV] += bytes;
    }

    return rc;
}

int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[],
                  void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        int rank, size;
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);

        size_t bytes = 0;
        for (int i=0; i<size; i++) {
            bytes += PLUMBER_count_dt_to_bytes(sendcounts[i], sendtypes[i]);
            bytes += PLUMBER_count_dt_to_bytes(recvcounts[i], recvtypes[i]);
        }

        plumber_commtype_count[ALLTOALLW] += 1;
        plumber_commtype_timer[ALLTOALLW] += (t1-t0);
        plumber_commtype_bytes[ALLTOALLW] += bytes;
    }

    return rc;
}

/* point-to-point communication */

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Send(buf, count, datatype, dest, tag, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[SEND] += 1;
        plumber_commtype_timer[SEND] += (t1-t0);
        plumber_commtype_bytes[SEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[BSEND] += 1;
        plumber_commtype_timer[BSEND] += (t1-t0);
        plumber_commtype_bytes[BSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[SSEND] += 1;
        plumber_commtype_timer[SSEND] += (t1-t0);
        plumber_commtype_bytes[SSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Rsend(buf, count, datatype, dest, tag, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[RSEND] += 1;
        plumber_commtype_timer[RSEND] += (t1-t0);
        plumber_commtype_bytes[RSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[ISEND] += 1;
        plumber_commtype_timer[ISEND] += (t1-t0);
        plumber_commtype_bytes[ISEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[IBSEND] += 1;
        plumber_commtype_timer[IBSEND] += (t1-t0);
        plumber_commtype_bytes[IBSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[ISSEND] += 1;
        plumber_commtype_timer[ISSEND] += (t1-t0);
        plumber_commtype_bytes[ISSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[IRSEND] += 1;
        plumber_commtype_timer[IRSEND] += (t1-t0);
        plumber_commtype_bytes[IRSEND] += bytes;

        if (plumber_sendmatrix_active) {
            plumber_sendmatrix_count[dest] += 1;
            plumber_sendmatrix_timer[dest] += (t1-t0);
            plumber_sendmatrix_bytes[dest] += bytes;
        }
    }

    return rc;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[RECV] += 1;
        plumber_commtype_timer[RECV] += (t1-t0);
        plumber_commtype_bytes[RECV] += bytes;
    }

    return rc;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[IRECV] += 1;
        plumber_commtype_timer[IRECV] += (t1-t0);
        plumber_commtype_bytes[IRECV] += bytes;
    }

    return rc;
}

int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Mrecv(buf, count, datatype, message, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[MRECV] += 1;
        plumber_commtype_timer[MRECV] += (t1-t0);
        plumber_commtype_bytes[MRECV] += bytes;
    }

    return rc;
}

int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Imrecv(buf, count, datatype, message, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);

        plumber_commtype_count[IMRECV] += 1;
        plumber_commtype_timer[IMRECV] += (t1-t0);
        plumber_commtype_bytes[IMRECV] += bytes;
    }

    return rc;
}

int PMPI_Wait(MPI_Request *request, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Wait(request, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[WAIT] += 1;
        plumber_commtype_timer[WAIT] += (t1-t0);
    }

    return rc;
}

int PMPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Test(request, flag, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[TEST] += 1;
        plumber_commtype_timer[TEST] += (t1-t0);
    }

    return rc;
}

int PMPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitany(count, requests, index, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[WAITANY] += 1;
        plumber_commtype_timer[WAITANY] += (t1-t0);
    }

    return rc;
}

int PMPI_Testany(int count, MPI_Request requests[], int *index, int *flag, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testany(count, requests, index, flag, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[TESTANY] += 1;
        plumber_commtype_timer[TESTANY] += (t1-t0);
    }

    return rc;
}

int PMPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitall(count, requests, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[WAITALL] += 1;
        plumber_commtype_timer[WAITALL] += (t1-t0);
    }

    return rc;
}

int PMPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testall(count, requests, flag, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[TESTALL] += 1;
        plumber_commtype_timer[TESTALL] += (t1-t0);
    }

    return rc;
}

int PMPI_Waitsome(int incount, MPI_Request requests[], int *outcount, int indices[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitsome(incount, requests, outcount, indices, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[WAITSOME] += 1;
        plumber_commtype_timer[WAITSOME] += (t1-t0);
    }

    return rc;
}

int PMPI_Testsome(int incount, MPI_Request requests[], int *outcount, int indices[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testsome(incount, requests, outcount, indices, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_commtype_count[TESTSOME] += 1;
        plumber_commtype_timer[TESTSOME] += (t1-t0);
    }

    return rc;
}

