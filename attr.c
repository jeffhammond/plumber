#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int copy(MPI_Comm oldcomm, int keyval, void * extra, void * attr_in, void * attr_out, int * flag)
{
    printf("copy of kv=%d attr_in=%d\n", keyval, *(int*)attr_in);
    return MPI_SUCCESS;
}

int delete(MPI_Comm oldcomm, int keyval, void * attr_val, void * extra)
{
    printf("delete of kv=%d attr_in=%d\n", keyval, *(int*)attr_val);
    return MPI_SUCCESS;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int kv;
    int state;
    MPI_Comm_create_keyval(copy, delete, &kv, &state);

    int wval = 0x86;
    MPI_Comm_set_attr(MPI_COMM_WORLD, kv, &wval);

    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);

    int oval, flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, kv, &oval, &flag);

    MPI_Comm_free(&dup);

    MPI_Comm_free_keyval(&kv);

    MPI_Finalize();
    return 0;
}
