/* 
 *
 * Implementación en C del "Algoritmo del Rey" usando MPI.
 * Simula nodos leales y nodos traidores (con fallos bizantinos).
 * Cada nodo intenta llegar a consenso entre "ATAQUE" o "RETIRADA".
 *
 * Autores:
 * 1. Sanchez Gonzalez Oscar Iván
 * 2. Barrera Hernández Tomás
 * Compilar:  /usr/lib64/openmpi/bin/mpicc rey_mpi.c -o rey_mpi
 * Ejecutar ejemplo: /usr/lib64/openmpi/bin/mpirun -np 4 --oversubscribe --mca btl_base_verbose -1 ./rey_mpi -f 1 -r 5
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define VAL_ATAQUE 1
#define VAL_RETIRADA 0

/* Función hash FNV-1a simple para obtener un valor numérico a partir de texto */
uint64_t fnv1a_hash(const unsigned char *data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

/* Convierte una lista de IDs en un string ordenado (para el cálculo del hash) */
char *ids_a_cadena_ordenada(int *ids, int n) {
    if (n == 0) {
        char *s = malloc(2);
        strcpy(s, "-");
        return s;
    }

    int *ordenados = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) ordenados[i] = ids[i];

    // Ordenamiento booblesort simple como en algoritmos xd
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (ordenados[i] > ordenados[j]) {
                int tmp = ordenados[i];
                ordenados[i] = ordenados[j];
                ordenados[j] = tmp;
            }

    // Crear cadena
    size_t tam = n * 10 + 4;
    char *s = malloc(tam);
    s[0] = '\0';
    char temp[16];
    for (int i = 0; i < n; ++i) {
        snprintf(temp, sizeof(temp), "%d,", ordenados[i]);
        strcat(s, temp);
    }
    free(ordenados);
    return s;
}

int elegir_rey_coordinado(int n, int *ids_union, int n_union, int rank) {
    int rey;
    
    if (rank == 0) {
        // Solo el nodo 0 calcula el rey de manera determinista
        long tiempo_comun = (long)time(NULL);
        // Hash basado en IDs de nodos que no alcanzaron consenso
        char *cadena = ids_a_cadena_ordenada(ids_union, n_union);
        uint64_t hash_base = fnv1a_hash((unsigned char*)cadena, strlen(cadena));
        free(cadena);
        // Combinar hash + tiempo del sistema y elegir nodo con valor más bajo
        uint64_t mejor_valor = UINT64_MAX;
        rey = 0;
        
        for (int id = 0; id < n; ++id) {
            uint64_t valor = hash_base + (uint64_t)tiempo_comun + (uint64_t)id;
            if (valor < mejor_valor) {
                mejor_valor = valor;
                rey = id;
            }
        }
    }
    
    // Compartir el rey con todos los nodos - GARANTIZA CONSISTENCIA
    MPI_Bcast(&rey, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return rey;
}

/* Lectura de argumentos desde línea de comandos */
void leer_argumentos(int argc, char **argv, int *f, int *max_rondas, unsigned long *semilla) {
    *f = -1;
    *max_rondas = 10;
    *semilla = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) *f = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) *max_rondas = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) *semilla = atoll(argv[++i]);
    }
}

/* Función principal */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    int f, max_rondas;
    unsigned long semilla;
    leer_argumentos(argc, argv, &f, &max_rondas, &semilla);
    if (f < 0) {
        if (rank == 0) fprintf(stderr, "Uso: %s -f <num_traidores> -r <max_rondas> [-s <semilla>]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Condición de seguridad: n > 3f
    if (!(n > 3 * f)) {
        if (rank == 0)
            fprintf(stderr, "ERROR: Se necesita n > 3f para tolerar traidores. n=%d f=%d\n", n, f);
        MPI_Finalize();
        return 1;
    }

    // Inicialización de semilla
    if (semilla == 0) semilla = (unsigned long)time(NULL) + rank * 17;
    srand((unsigned int)(semilla + rank * 11));

    // Nodo 0 elige quiénes serán traidores
    int *traidores = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) traidores[i] = 0;

    if (rank == 0) {
        int *ids = malloc(n * sizeof(int));
        for (int i = 0; i < n; ++i) ids[i] = i;

        // Barajar y elegir los primeros f como traidores
        for (int i = n - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int tmp = ids[i];
            ids[i] = ids[j];
            ids[j] = tmp;
        }
        for (int i = 0; i < f; ++i)
            traidores[ids[i]] = 1;

        free(ids);
    }

    // Compartir lista de traidores con todos los nodos
    MPI_Bcast(traidores, n, MPI_INT, 0, MPI_COMM_WORLD);
    int soy_traidor = traidores[rank];

    // Plan inicial: cada nodo elige ataque o retirads al azar
    int mi_valor;
    if (soy_traidor) {
        // Traidores tienen preferencia por ATAQUE para crear conflicto
        mi_valor = (rand() % 100 < 70) ? VAL_ATAQUE : VAL_RETIRADA;
    } else {
        // Leales más balanceados
        mi_valor = (rand() % 100 < 50) ? VAL_ATAQUE : VAL_RETIRADA;  // 50/50 exacto
    }
    int valor_final = mi_valor;

    if (rank == 0) {
        printf("Nodos totales: %d | Traidores: %d | Máx. rondas: %d\n", n, f, max_rondas);
        printf("Lista de traidores (1=traidor): ");
        for (int i = 0; i < n; ++i) printf("%d", traidores[i]);
        printf("\n\n");
    }

    int *recibidos = malloc(n * sizeof(int));
    MPI_Status estado;
    int consenso = 0;

    for (int ronda = 1; ronda <= max_rondas && !consenso; ++ronda) {
        ronda = ronda;
        // Comunicación entre todos los nodos
        for (int dest = 0; dest < n; ++dest) {
            if (dest == rank) continue;
            int enviar;
            
            if (soy_traidor)
                enviar = rand() % 2;  // Traidor manda valores aleatorios
            else
                enviar = mi_valor;

            MPI_Send(&enviar, 1, MPI_INT, dest, ronda, MPI_COMM_WORLD);
        }

        // Recibir los valores de los demás nodos
        recibidos[rank] = mi_valor;
        for (int src = 0; src < n; ++src) {
            if (src == rank) continue;
            MPI_Recv(&recibidos[src], 1, MPI_INT, src, ronda, MPI_COMM_WORLD, &estado);
        }

        // Contar votos
        int cuenta_ataque = 0, cuenta_retirada = 0;
        for (int i = 0; i < n; ++i) {
            if (recibidos[i] == VAL_ATAQUE) cuenta_ataque++;
            else cuenta_retirada++;
        }

        int mayoria_valor = (cuenta_ataque > cuenta_retirada) ? VAL_ATAQUE : VAL_RETIRADA;
        int mayoria_cuenta = (cuenta_ataque > cuenta_retirada) ? cuenta_ataque : cuenta_retirada;

        // Nodos que no coinciden con la mayoría
        int *no_consenso = malloc(n * sizeof(int));
        int n_no = 0;
        for (int i = 0; i < n; ++i)
            if (recibidos[i] != mayoria_valor) no_consenso[n_no++] = i;

        // Crear vector binario de presencia y combinar con todos (MPI_Allreduce)
        int *presencia = calloc(n, sizeof(int));
        for (int i = 0; i < n_no; ++i) presencia[no_consenso[i]] = 1;
        int *global_presencia = malloc(n * sizeof(int));
        MPI_Allreduce(presencia, global_presencia, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Generar lista de unión de nodos sin consenso
        int *union_ids = malloc(n * sizeof(int));
        int n_union = 0;
        for (int i = 0; i < n; ++i)
            if (global_presencia[i]) union_ids[n_union++] = i;

        //Elegir al rey después de cada ronda sin consenso
        int rey = elegir_rey_coordinado(n, union_ids, n_union, rank);

        int valor_rey;
        if (rank == rey) {
            if (soy_traidor) {
                // Rey traidor elige estratégicamente
                valor_rey = rand() % 2;  // Información arbitraria
            } else {
                valor_rey = mayoria_valor;
            }
        }


        if (rank == rey) {
            for (int dest = 0; dest < n; ++dest)
                if (dest != rank)
                    MPI_Send(&valor_rey, 1, MPI_INT, dest, 1000 + ronda, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&valor_rey, 1, MPI_INT, rey, 1000 + ronda, MPI_COMM_WORLD, &estado);
        }

        // Regla de decisión: mantener mayoría o seguir al rey
        if (mayoria_cuenta > f)
            mi_valor = mayoria_valor;
        else
            mi_valor = valor_rey;

        // Comprobar si ya todos tienen el mismo valor
        int *todos = malloc(n * sizeof(int));
        MPI_Allgather(&mi_valor, 1, MPI_INT, todos, 1, MPI_INT, MPI_COMM_WORLD);

        int iguales = 1;
        for (int i = 0; i < n; ++i)
            if (todos[i] != todos[0]) { iguales = 0; break; }

        if (iguales) {
            consenso = 1;
            valor_final = mi_valor;
            if (rank == 0)
                printf(" Ronda %d: CONSENSO alcanzado → %s\n",
                       ronda, valor_final == VAL_ATAQUE ? "ATAQUE" : "RETIRADA");
        } else if (rank == 0) {
            printf("  Ronda %d: Sin consenso. Rey elegido: %d (Votos: Ataque=%d, Retirada=%d)\n", 
                   ronda, rey, cuenta_ataque, cuenta_retirada);
        }

        free(no_consenso);
        free(presencia);
        free(global_presencia);
        free(union_ids);
        free(todos);
    }

    // REQUISITO 4: Reportar resultado final
    MPI_Barrier(MPI_COMM_WORLD);
    if (!consenso && rank == 0) {
        printf("✗ Máximo de rondas (%d) alcanzado SIN CONSENSO\n", max_rondas);
    }

    // ---- Reporte final por nodo ----
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Nodo %d | Traidor=%d | Valor final=%s\n",
           rank, soy_traidor, valor_final == VAL_ATAQUE ? "ATAQUE" : "RETIRADA");

    free(traidores);
    free(recibidos);
    MPI_Finalize();
    return 0;
}
