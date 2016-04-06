#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include <mpi.h>

/* because i do not know how to use fixed-width
 * printf with PRIu64 */
typedef unsigned long long int myu64_t;

/********************************************
 * internal data
 ********************************************/

int plumber_profiling_active = 0;
int plumber_p2pmatrix_active = 0;
int plumber_rmamatrix_active = 0;
int plumber_subcomm_profiling = 0;

/* if non-zero (1), mutual exclusion is required */
int plumber_multithreaded = 0;
#ifdef HAVE_PTHREAD_H
/* Pthread implemenation */
typedef pthread_mutex_t plumber_mutex_t;
plumber_mutex_t plumber_mutex = PTHREAD_MUTEX_INITIALIZER;
static inline int plumber_mutex_lock(plumber_mutex_t * mutex) { return pthread_mutex_lock(mutex); }
static inline int plumber_mutex_unlock(plumber_mutex_t * mutex) { return pthread_mutex_unlock(mutex); }
#else
/* dummy (impotent) implementation */
#warning This mutex will not ensure thread-safety
typedef pthread_mutex_t int;
plumber_mutex_t plumber_mutex = 0;
static inline int plumber_mutex_lock(plumber_mutex_t * mutex) { *mutex = 1; return 0; }
static inline int plumber_mutex_unlock(plumber_mutex_t * mutex) { *mutex = 0; return 0; }
#endif

/* capture these in init and use in finalize */
int plumber_argc;
char** plumber_argv;

/* caching data on subcomms */
int plumber_comm_keyval;

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
    /* RMA */
    FETCHOP       = 34,
    COMPSWAP      = 35,
    ACC           = 36,
    GET           = 37,
    PUT           = 38,
    GETACC        = 39,
    RACC          = 40,
    RGET          = 41,
    RPUT          = 42,
    RGETACC       = 43,
    /* the end */
    MAX_COMMTYPE  = 44
} plumber_commtype_t;

char plumber_commtype_names[MAX_COMMTYPE][21] = {
"Send",
"Bsend",
"Ssend",
"Rsend",
"Isend",
"Ibsend",
"Issend",
"Irsend",
"Recv",
"Irecv",
"Mrecv",
"Imrecv",
"Bcast",
"Reduce",
"Allreduce",
"Alltoall",
"Alltoallv",
"Gather",
"Allgather",
"Scatter",
"Gatherv",
"Allgatherv",
"Scatterv",
"Reduce_scatter",
"Reduce_scatter_block",
"Alltoallw",
"Fetch_and_op",
"Compare_and_swap",
"Accumulate",
"Get",
"Put",
"Get_accumulate"
};

typedef enum {
    WAIT             = 0,
    WAITANY          = 1,
    WAITSOME         = 2,
    WAITALL          = 3,
    TEST             = 4,
    TESTANY          = 5,
    TESTSOME         = 6,
    TESTALL          = 7,
    BARRIER          = 8,
    COMMDUP          = 9,
    COMMCREATE       = 10,
    COMMSPLIT        = 11,
    COMMSPLITTYPE    = 12,
    COMMFREE         = 13,
    WINCREATE        = 14,
    WINALLOC         = 15,
    WINALLOCSH       = 16,
    WINCREATEDYN     = 17,
    WINATTACH        = 18,
    WINDETACH        = 19,
    WINFREE          = 20,
    WINFENCE         = 21,
    WINSYNC          = 22,
    WINLOCK          = 23,
    WINUNLOCK        = 24,
    WINLOCKALL       = 25,
    WINUNLOCKALL     = 27,
    WINFLUSH         = 28,
    WINFLUSHALL      = 29,
    WINFLUSHLOCAL    = 30,
    WINFLUSHLOCALALL = 31,
    WINPOST          = 32,
    WINSTART         = 33,
    WINCOMPLETE      = 34,
    WINWAIT          = 35,
    WINTEST          = 36,
    /* the end */
    MAX_UTILTYPE     = 37
} plumber_utiltype_t;

char plumber_utiltype_names[MAX_UTILTYPE][20] = {
"Wait",
"Waitany",
"Waitsome",
"Waitall",
"Test",
"Testany",
"Testsome",
"Testall",
"Barrier",
"Comm_dup",
"Comm_create",
"Comm_split",
"Comm_split_type",
"Comm_free",
"Win_create",
"Win_allocate",
"Win_allocate_shared",
"Win_create_dynamic",
"Win_attach",
"Win_detach",
"Win_free",
"Win_fence",
"Win_sync",
"Win_lock",
"Win_unlock",
"Win_lock_all",
"Win_unlock_all",
"Win_flush",
"Win_flush_all",
"Win_flush_local",
"Win_flush_local_all",
"Win_post",
"Win_start",
"Win_complete",
"Win_wait",
"Win_test"
};

myu64_t plumber_commtype_count[MAX_COMMTYPE] = {0};
double  plumber_commtype_timer[MAX_COMMTYPE] = {0};
myu64_t plumber_commtype_bytes[MAX_COMMTYPE] = {0};

myu64_t plumber_utiltype_count[MAX_UTILTYPE] = {0};
double  plumber_utiltype_timer[MAX_UTILTYPE] = {0};

/* dynamically allocated due to O(nproc) */
myu64_t * plumber_p2pmatrix_count = NULL;
double  * plumber_p2pmatrix_timer = NULL;
myu64_t * plumber_p2pmatrix_bytes = NULL;

/* dynamically allocated due to O(nproc) */
myu64_t * plumber_rmamatrix_count = NULL;
double  * plumber_rmamatrix_timer = NULL;
myu64_t * plumber_rmamatrix_bytes = NULL;

/* time from the end of PLUMBER_init to capture application time */
double plumber_start_time = 0.0;

/* this is the state used for profiling on user-defined communicators */
typedef struct {
    myu64_t commtype_count[MAX_COMMTYPE];
    double  commtype_timer[MAX_COMMTYPE];
    myu64_t commtype_bytes[MAX_COMMTYPE];

    myu64_t utiltype_count[MAX_UTILTYPE];
    double  utiltype_timer[MAX_UTILTYPE];

    myu64_t * p2pmatrix_count;
    double  * p2pmatrix_timer;
    myu64_t * p2pmatrix_bytes;

#if 0
    myu64_t * rmamatrix_count;
    double  * rmamatrix_timer;
    myu64_t * rmamatrix_bytes;
#endif

    double start_time;
} plumber_usercomm_data_t;

/********************************************
 * internal functions
 ********************************************/

static inline void PLUMBER_add2(myu64_t * o1, double * o2,
                                myu64_t   i1, double   i2)
{
    if (plumber_multithreaded) plumber_mutex_lock(&plumber_mutex);
    {
        *o1 += i1;
        *o2 += i2;
    }
    if (plumber_multithreaded) plumber_mutex_unlock(&plumber_mutex);
}

static inline void PLUMBER_add3(myu64_t * o1, double * o2, myu64_t * o3,
                                myu64_t   i1, double   i2, double     i3)
{
    if (plumber_multithreaded) plumber_mutex_lock(&plumber_mutex);
    {
        *o1 += i1;
        *o2 += i2;
        *o3 += i3;
    }
    if (plumber_multithreaded) plumber_mutex_unlock(&plumber_mutex);
}

/* replace with more accurate timer if necessary */
static double PLUMBER_wtime(void)
{
    return ( plumber_profiling_active ? PMPI_Wtime() : 0.0 );
}

static int PLUMBER_init_comm_matrix(MPI_Comm comm, myu64_t ** count, double ** timer, myu64_t ** bytes)
{
    int size;
    PMPI_Comm_size(comm, &size);

    *count = malloc(size * sizeof(myu64_t));
    *timer = malloc(size * sizeof(double));
    *bytes = malloc(size * sizeof(myu64_t));

    if (*count == NULL || *timer == NULL || *bytes == NULL) {
        fprintf(stderr, "PLUMBER: communication matrix memory allocation did not succeed for %d processes\n", size);
        return 1;
    }

    for (int i=0; i<size; i++) {
        (*count)[i] = 0;
        (*timer)[i] = 0.0;
        (*bytes)[i] = 0;
    }

    return 0;
}

static int PLUMBER_init_usercomm_data(MPI_Comm comm, plumber_usercomm_data_t * data)
{
    for (int i=0; i<MAX_COMMTYPE; i++) {
        data->commtype_count[i] = 0;
        data->commtype_timer[i] = 0.0;
        data->commtype_bytes[i] = 0;
    }
    for (int i=0; i<MAX_UTILTYPE; i++) {
        data->utiltype_count[i] = 0;
        data->utiltype_timer[i] = 0.0;
    }
    int rc = PLUMBER_init_comm_matrix(comm, &(data->p2pmatrix_count),
                                            &(data->p2pmatrix_timer),
                                            &(data->p2pmatrix_bytes));
    return rc;
}

static int PLUMBER_finalize_comm_matrix(MPI_Comm comm, myu64_t * count, double * timer, myu64_t * bytes,
                                        char* filepath, char* matrix_name)
{
    int size, rank;
    PMPI_Comm_size(comm, &size);
    PMPI_Comm_rank(comm, &rank);

    FILE * file = fopen(filepath, "w");
    if ( file==NULL ) {
        fprintf(stderr, "PLUMBER: fopen of %s did not succeed\n", filepath);
        return 1;
    } else {
        char name[MPI_MAX_OBJECT_NAME] = {0};
        int len;
        PMPI_Comm_get_name(comm, name, &len);

        fprintf(file, "PLUMBER %s matrix for process %d\n", matrix_name, rank);
        fprintf(file, "communicator size = %d, name = %s\n", size, name);
        fprintf(file, "%10s %10s %30s %20s\n", "target", "calls", "time", "bytes");
        for (int i=0; i<size; i++) {
            if (count[i] > 0) {
                fprintf(file, "%10d %20llu %30.14lf %20llu\n", i, count[i], timer[i], bytes[i]);
            }
        }
        fprintf(file, "EOF\n");
        fclose(file);
    }

    free(count);
    free(timer);
    free(bytes);

    return 0;
}

static void PLUMBER_init(int argc, char** argv, int threading)
{
    /* fixme */
    plumber_profiling_active = 1;

    /* no mutex required because MPI_Init(_thread) can only be called
     * from one thread. */
    if (plumber_profiling_active) {

        plumber_multithreaded = (threading==MPI_THREAD_MULTIPLE) ? 1 : 0;
        if (plumber_multithreaded) {

        }

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

        /* fixme */
        plumber_p2pmatrix_active = 1;

        if (plumber_p2pmatrix_active) {
            int rc = PLUMBER_init_comm_matrix( MPI_COMM_WORLD,
                                               &plumber_p2pmatrix_count,
                                               &plumber_p2pmatrix_timer,
                                               &plumber_p2pmatrix_bytes);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_comm_matrix failed\n");
            }
        }

        /* fixme */
        plumber_rmamatrix_active = 0;

        if (plumber_rmamatrix_active) {
            int rc = PLUMBER_init_comm_matrix( MPI_COMM_WORLD,
                                               &plumber_rmamatrix_count,
                                               &plumber_rmamatrix_timer,
                                               &plumber_rmamatrix_bytes);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_comm_matrix failed\n");
            }
        }

        /* fixme */
        plumber_subcomm_profiling = 1;
        if (plumber_subcomm_profiling) {
            MPI_Comm_copy_attr_function   * plumber_comm_copy_attr_fn   = MPI_COMM_NULL_COPY_FN;
            MPI_Comm_delete_attr_function * plumber_comm_delete_attr_fn = MPI_COMM_NULL_DELETE_FN;
            PMPI_Comm_create_keyval(plumber_comm_copy_attr_fn, plumber_comm_delete_attr_fn, &plumber_comm_keyval, NULL);
        }

        plumber_start_time = PLUMBER_wtime();
    }
}

static void PLUMBER_finalize(int collective)
{
    /* no mutex required because MPI_Finalize can only be called
     * from one thread. */
    if (plumber_profiling_active) {

        double plumber_end_time = PLUMBER_wtime();
        double plumber_app_time = plumber_end_time - plumber_start_time;

        int rank, size;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);

        char summaryfilepath[256];
        char rankfilepath[256];
        char p2pmatrixfilepath[256];
        char rmamatrixfilepath[256];

        char * prefix = getenv("PLUMBER_PREFIX");
        if (prefix != NULL) {
            strncpy(summaryfilepath,   prefix, 255);
            strncpy(rankfilepath,      prefix, 255);
            strncpy(p2pmatrixfilepath, prefix, 255);
            strncpy(rmamatrixfilepath, prefix, 255);
        } else {
            char plumber_program_name[255];
            if (plumber_argc>0) {
                strncpy(plumber_program_name, plumber_argv[0], 255);
            } else {
                strncpy(plumber_program_name, "unknown", 255);
            }
            /* append plumber_program_name with timestamp to be unique... */
            strncpy(summaryfilepath,   plumber_program_name, 255);
            strncpy(rankfilepath,      plumber_program_name, 255);
            strncpy(p2pmatrixfilepath, plumber_program_name, 255);
            strncpy(rmamatrixfilepath, plumber_program_name, 255);
        }

        /* 2^31 = 2147483648 requires 10 digits */
        char rankstring[12] = {0};
        sprintf(rankstring, "%d", rank);

        strcat(summaryfilepath, ".plumber.summary.");
        strcat(summaryfilepath, rankstring);

        strcat(rankfilepath, ".plumber.profile.");
        strcat(rankfilepath, rankstring);

        strcat(p2pmatrixfilepath, ".plumber.matrix.");
        strcat(p2pmatrixfilepath, rankstring);

        strcat(rmamatrixfilepath, ".plumber.rmamatrix.");
        strcat(rmamatrixfilepath, rankstring);

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
            fprintf(rankfile, "total MPI time = %lf (%6.2lf percent)\n",
                              plumber_total_mpi_time,
                              100.*plumber_total_mpi_time/plumber_app_time);
            /* MPI profile */
            fprintf(rankfile, "%22s %20s %30s %20s\n", "function", "calls", "time", "bytes");
            for (int i=0; i<MAX_COMMTYPE; i++) {
                if (plumber_commtype_count[i] > 0) {
                    fprintf(rankfile, "%22s %20llu %30.14lf %20llu\n",
                            plumber_commtype_names[i],
                            plumber_commtype_count[i],
                            plumber_commtype_timer[i],
                            plumber_commtype_bytes[i]);
                }
            }
            for (int i=0; i<MAX_UTILTYPE; i++) {
                if (plumber_utiltype_count[i] > 0) {
                    fprintf(rankfile, "%22s %20llu %30.14lf\n",
                            plumber_utiltype_names[i],
                            plumber_utiltype_count[i],
                            plumber_utiltype_timer[i]);
                }
            }
            fprintf(rankfile, "EOF\n");
            fclose(rankfile);
        } /* rankfile fopen success */

        if (plumber_p2pmatrix_active) {
            PLUMBER_finalize_comm_matrix( MPI_COMM_WORLD,
                                          plumber_p2pmatrix_count,
                                          plumber_p2pmatrix_timer,
                                          plumber_p2pmatrix_bytes,
                                          p2pmatrixfilepath, "p2p");
        }

        if (plumber_rmamatrix_active) {
            PLUMBER_finalize_comm_matrix( MPI_COMM_WORLD,
                                          plumber_rmamatrix_count,
                                          plumber_rmamatrix_timer,
                                          plumber_rmamatrix_bytes,
                                          rmamatrixfilepath, "rma");
        }

        if (collective) {
            /* reduce to get totals */
            myu64_t total_commtype_count[MAX_COMMTYPE];
            double   total_commtype_timer[MAX_COMMTYPE];
            myu64_t total_commtype_bytes[MAX_COMMTYPE];
            myu64_t total_utiltype_count[MAX_UTILTYPE];
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
                    fprintf(rankfile, "%22s %20s %30s %20s\n", "function", "calls", "time", "bytes");
                    for (int i=0; i<MAX_COMMTYPE; i++) {
                        if (total_commtype_count[i] > 0) {
                            fprintf(rankfile, "%22s %20llu %30.14lf %20llu\n",
                                    plumber_commtype_names[i],
                                    total_commtype_count[i],
                                    total_commtype_timer[i],
                                    total_commtype_bytes[i]);
                        }
                    }
                    for (int i=0; i<MAX_UTILTYPE; i++) {
                        if (total_utiltype_count[i] > 0) {
                            fprintf(rankfile, "%22s %20llu %30.14lf\n",
                                    plumber_utiltype_names[i],
                                    total_utiltype_count[i],
                                    total_utiltype_timer[i]);
                        }
                    }
                    fprintf(summaryfile, "EOF\n");
                    fclose(summaryfile);
                }
            } /* rank==0 */
        } /* collective */

        if (plumber_subcomm_profiling) {
            PMPI_Comm_free_keyval(&plumber_comm_keyval);
        }

    } /* plumber_profiling_active */
}

/* this does not yet support user-defined datatypes */
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
        PLUMBER_init(*argc, *argv, MPI_THREAD_SINGLE);
    } else {
        PLUMBER_init(0, NULL, MPI_THREAD_SINGLE);
    }
    return rc;
}

int MPI_Init_thread(int * argc, char** * argv, int requested, int * provided)
{
    int rc = PMPI_Init_thread(argc, argv, requested, provided);
    if (argc != NULL && argv != NULL) {
        PLUMBER_init(*argc, *argv, *provided);
    } else {
        PLUMBER_init(0, NULL, *provided);
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
        plumber_utiltype_t offset = BARRIER;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_dup(comm, newcomm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMDUP;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

        if (plumber_subcomm_profiling) {
            plumber_usercomm_data_t * ptr = malloc(sizeof(plumber_usercomm_data_t));
            int rc = PLUMBER_init_usercomm_data(*newcomm, ptr);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_usercomm_data failed\n");
            }
            PMPI_Comm_set_attr(*newcomm, plumber_comm_keyval, ptr);
        }
    }
    return rc;
}

int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_dup_with_info(comm, info, newcomm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMDUP;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

        if (plumber_subcomm_profiling) {
            plumber_usercomm_data_t * ptr = malloc(sizeof(plumber_usercomm_data_t));
            int rc = PLUMBER_init_usercomm_data(*newcomm, ptr);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_usercomm_data failed\n");
            }
            PMPI_Comm_set_attr(*newcomm, plumber_comm_keyval, ptr);
        }
    }
    return rc;
}

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_create(comm, group, newcomm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMCREATE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

        if (plumber_subcomm_profiling) {
            plumber_usercomm_data_t * ptr = malloc(sizeof(plumber_usercomm_data_t));
            int rc = PLUMBER_init_usercomm_data(*newcomm, ptr);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_usercomm_data failed\n");
            }
            PMPI_Comm_set_attr(*newcomm, plumber_comm_keyval, ptr);
        }
    }
    return rc;
}

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMSPLITTYPE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

        if (plumber_subcomm_profiling) {
            plumber_usercomm_data_t * ptr = malloc(sizeof(plumber_usercomm_data_t));
            int rc = PLUMBER_init_usercomm_data(*newcomm, ptr);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_usercomm_data failed\n");
            }
            PMPI_Comm_set_attr(*newcomm, plumber_comm_keyval, ptr);
        }
    }
    return rc;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_split(comm, color, key, newcomm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMSPLIT;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

        if (plumber_subcomm_profiling) {
            plumber_usercomm_data_t * ptr = malloc(sizeof(plumber_usercomm_data_t));
            int rc = PLUMBER_init_usercomm_data(*newcomm, ptr);
            if (rc) {
                fprintf(stderr, "PLUMBER_init_usercomm_data failed\n");
            }
            PMPI_Comm_set_attr(*newcomm, plumber_comm_keyval, ptr);
        }
    }
    return rc;
}

int MPI_Comm_free(MPI_Comm *comm)
{
    if (plumber_profiling_active) {
        if (plumber_subcomm_profiling) {
            int flag;
            plumber_usercomm_data_t * ptr;
            PMPI_Comm_get_attr(*comm, plumber_comm_keyval, &ptr, &flag);
            if (!flag) {
                fprintf(stderr, "PMPI_Comm_get_attr flag=%d\n", flag);
            }

            char p2pmatrixfilepath[255];
            char * prefix = getenv("PLUMBER_PREFIX");
            if (prefix != NULL) {
                strncpy(p2pmatrixfilepath, prefix, 255);
            } else {
                char plumber_program_name[255];
                if (plumber_argc>0) {
                    strncpy(plumber_program_name, plumber_argv[0], 255);
                } else {
                    strncpy(plumber_program_name, "unknown", 255);
                }
                strncpy(p2pmatrixfilepath, plumber_program_name, 255);
            }
            strcat(p2pmatrixfilepath, ".plumber.matrix.");
            /* 2^31 = 2147483648 requires 10 digits */
            int rank;
            PMPI_Comm_rank(*comm, &rank);
            char rankstring[12] = {0};
            sprintf(rankstring, "%d", rank);
            strcat(p2pmatrixfilepath, rankstring);

            char name[MPI_MAX_OBJECT_NAME] = {0};
            int len;
            PMPI_Comm_get_name(*comm, name, &len);
            if (!len) {
                strncpy(name, "noname", MPI_MAX_OBJECT_NAME-1);
            }
            strcat(p2pmatrixfilepath, ".");
            strcat(p2pmatrixfilepath, name);

            PLUMBER_finalize_comm_matrix( *comm,
                                          ptr->p2pmatrix_count,
                                          ptr->p2pmatrix_timer,
                                          ptr->p2pmatrix_bytes,
                                          p2pmatrixfilepath, "p2p");
            free(ptr);
        }
    }

    double t0 = PLUMBER_wtime();
    int rc = PMPI_Comm_free(comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = COMMFREE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);

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
        plumber_commtype_t offset = REDUCE;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLREDUCE;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = REDSCAT;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = REDSCATB;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = BCAST;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = GATHER;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = GATHERV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = SCATTER;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = SCATTERV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLGATHER;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLGATHERV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLTOALL;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLTOALLV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = ALLTOALLW;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
    }

    return rc;
}

/* point-to-point communication */

static void PLUMBER_p2p_capture(plumber_commtype_t offset, double dt,
                                int count, MPI_Datatype datatype, int dest, MPI_Comm comm)
{
    size_t bytes = PLUMBER_count_dt_to_bytes(count, datatype);
    PLUMBER_add3( &plumber_commtype_count[offset],
                  &plumber_commtype_timer[offset],
                  &plumber_commtype_bytes[offset],
                  1, dt, bytes);

    if (plumber_p2pmatrix_active) {
        plumber_p2pmatrix_count[dest] += 1;
        plumber_p2pmatrix_timer[dest] += dt;
        plumber_p2pmatrix_bytes[dest] += bytes;
    }

    if (plumber_subcomm_profiling) {
        int flag;
        plumber_usercomm_data_t * ptr;
        PMPI_Comm_get_attr(comm, plumber_comm_keyval, &ptr, &flag);
        if (!flag) {
            fprintf(stderr, "PMPI_Comm_get_attr flag=%d\n", flag);
        }
    }
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Send(buf, count, datatype, dest, tag, comm);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        PLUMBER_p2p_capture(SEND, t1-t0, count, datatype, dest, comm);
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
        plumber_commtype_t offset = BSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = SSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = RSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = ISEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = IBSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = ISSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = IRSEND;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_p2pmatrix_active) {
            plumber_p2pmatrix_count[dest] += 1;
            plumber_p2pmatrix_timer[dest] += (t1-t0);
            plumber_p2pmatrix_bytes[dest] += bytes;
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
        plumber_commtype_t offset = RECV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = IRECV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = MRECV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
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
        plumber_commtype_t offset = IMRECV;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);
    }

    return rc;
}

int PMPI_Wait(MPI_Request *request, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Wait(request, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WAIT;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Test(request, flag, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = TEST;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitany(count, requests, index, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WAITANY;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Testany(int count, MPI_Request requests[], int *index, int *flag, MPI_Status *status)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testany(count, requests, index, flag, status);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = TESTANY;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitall(count, requests, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WAITALL;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testall(count, requests, flag, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = TESTALL;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Waitsome(int incount, MPI_Request requests[], int *outcount, int indices[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Waitsome(incount, requests, outcount, indices, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WAITSOME;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

int PMPI_Testsome(int incount, MPI_Request requests[], int *outcount, int indices[], MPI_Status statuses[])
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Testsome(incount, requests, outcount, indices, statuses);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        plumber_utiltype_t offset = TESTSOME;
        PLUMBER_add2( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      1, t1-t0);
    }

    return rc;
}

/* one-sided communication */

/* window ctor+dtor */
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_create(base, size, disp_unit, info, comm, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINCREATE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINALLOC;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINALLOCSH;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_create_dynamic(info, comm, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINCREATEDYN;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_attach(win, base, size);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINATTACH;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_detach(MPI_Win win, const void *base)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_detach(win, base);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINDETACH;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_free(MPI_Win *win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_free(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFREE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}


/* data movement */
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                     MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                     MPI_Op op, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Fetch_and_op(origin_addr, result_addr, datatype, target_rank, target_disp, op, win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(1, datatype);
        plumber_commtype_t offset = FETCHOP;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                         void *result_addr, MPI_Datatype datatype, int target_rank,
                         MPI_Aint target_disp, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(1, datatype);
        plumber_commtype_t offset = COMPSWAP;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
                   MPI_Op op, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Accumulate(origin_addr, origin_count, origin_datatype,
                             target_rank, target_disp, target_count, target_datatype,
                             op, win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = ACC;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
            MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Get(origin_addr, origin_count, origin_datatype,
                      target_rank, target_disp, target_count, target_datatype,
                      win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = GET;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
            MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Put(origin_addr, origin_count, origin_datatype,
                      target_rank, target_disp, target_count, target_datatype,
                      win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = PUT;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Get_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                       void *result_addr, int result_count, MPI_Datatype result_datatype,
                       int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
                       MPI_Op op, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Get_accumulate(origin_addr, origin_count, origin_datatype,
                                 result_addr, result_count, result_datatype,
                                 target_rank, target_disp, target_count, target_datatype,
                                 op, win);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        /* if and only if we are bringing data back do we count the return trip bytes */
        if (op != MPI_NO_OP) {
            bytes = PLUMBER_count_dt_to_bytes(result_count, result_datatype);
        }
        plumber_commtype_t offset = GETACC;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                    int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
                    MPI_Op op, MPI_Win win, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Raccumulate(origin_addr, origin_count, origin_datatype,
                              target_rank, target_disp, target_count, target_datatype,
                              op, win, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = RACC;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Rput(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
             MPI_Win win, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Rput(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype,
                       win, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = RPUT;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Rget(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp, int target_count,
             MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Rget(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype,
                       win, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        plumber_commtype_t offset = RGET;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

int MPI_Rget_accumulate(const void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, void *result_addr, int result_count,
                        MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                        MPI_Request *request)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Rget_accumulate(origin_addr, origin_count, origin_datatype,
                                  result_addr, result_count, result_datatype,
                                  target_rank, target_disp, target_count, target_datatype,
                                  op, win, request);
    double t1 = PLUMBER_wtime();

    if (plumber_profiling_active) {
        size_t bytes = PLUMBER_count_dt_to_bytes(origin_count, origin_datatype);
        /* if and only if we are bringing data back do we count the return trip bytes */
        if (op != MPI_NO_OP) {
            bytes = PLUMBER_count_dt_to_bytes(result_count, result_datatype);
        }
        plumber_commtype_t offset = RGETACC;
        PLUMBER_add3( &plumber_commtype_count[offset],
                      &plumber_commtype_timer[offset],
                      &plumber_commtype_bytes[offset],
                      1, t1-t0, bytes);

        if (plumber_rmamatrix_active) {
            plumber_rmamatrix_count[target_rank] += 1;
            plumber_rmamatrix_timer[target_rank] += (t1-t0);
            plumber_rmamatrix_bytes[target_rank] += bytes;
        }
    }

    return rc;
}

/* BSP */
int MPI_Win_fence(int assert, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_fence(assert, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFENCE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

/* PASSIVE TARGET */
int MPI_Win_sync(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_sync(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINSYNC;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_lock(lock_type, rank, assert, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINLOCK;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_unlock(int rank, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_unlock(rank, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINUNLOCK;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_lock_all(int assert, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_lock_all(assert, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINLOCKALL;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_unlock_all(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_unlock_all(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINUNLOCKALL;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_flush(int rank, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_flush(rank, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFLUSH;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_flush_all(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_flush_all(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFLUSHALL;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_flush_local(int rank, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_flush_local(rank, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFLUSHLOCAL;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_flush_local_all(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_flush_local_all(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINFLUSHLOCALALL;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

/* PSCW */
int MPI_Win_post(MPI_Group group, int assert, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_post(group, assert, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINPOST;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_start(MPI_Group group, int assert, MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_start(group, assert, win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINSTART;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_complete(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_complete(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINCOMPLETE;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_wait(MPI_Win win)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_wait(win);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINWAIT;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

int MPI_Win_test(MPI_Win win, int *flag)
{
    double t0 = PLUMBER_wtime();
    int rc = PMPI_Win_test(win, flag);
    double t1 = PLUMBER_wtime();
    if (plumber_profiling_active) {
        plumber_utiltype_t offset = WINTEST;
        PLUMBER_add2( &plumber_utiltype_count[offset],
                      &plumber_utiltype_timer[offset],
                      1, t1-t0);
    }
    return rc;
}

